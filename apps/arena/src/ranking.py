"""Bradley-Terry pairwise ranking for the synth arena (Phase 8).

`rank(config, api_key, sink)` runs `rounds x scenarios` race instances
per profile (all profiles racing each scenario together), scores each
final WorldState, builds a pairwise win matrix, and solves for
Bradley-Terry log-ratings via iterative Minorization-Maximization.

The 95% confidence interval is computed by percentile bootstrap over
the pairwise outcomes. `bootstrap_seed` makes the result reproducible.

`rank_with_mock()` mirrors `rank()` but uses the offline deterministic
invoke from `arena.race.race_with_mock` — for tests and CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from arena.config_profile import RaceConfig
from arena.race import VariantResult, race, race_with_mock
from arena.scenarios import DEFAULT_SCENARIOS, get_scenario
from evaluation.event_log import Event, MetricPayload, NullEventSink
from evaluation.world_sim import AGE_SEQUENCE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from arena.config_profile import RankingConfig
    from arena.scenarios import Scenario
    from evaluation.event_log import EventSink
    from evaluation.world_sim import WorldState
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ScoreTuple: TypeAlias = tuple[int, int, float]
ScoreFn: TypeAlias = "Callable[[WorldState], ScoreTuple]"
RaceFn: TypeAlias = "Callable[[RaceConfig, WorldState, EventSink], Awaitable[list[VariantResult]]]"
_F64Array: TypeAlias = "NDArray[np.float64]"


class RankingError(Exception):
    """Raised when the BT solver fails to converge."""


@dataclass(frozen=True, slots=True)
class PairwiseOutcome:
    """One race-instance result; `score` is the higher-wins composite."""

    profile_name: str
    scenario_name: str
    round_idx: int
    score: ScoreTuple


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Final Bradley-Terry ratings + 95% CIs."""

    ratings: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    pairwise_wins: dict[tuple[str, str], int]
    n_races: int


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def composite_score(state: WorldState) -> ScoreTuple:
    """Lexicographic score: age first, then population, then food + wood."""
    return (
        AGE_SEQUENCE.index(state.age),
        state.population,
        state.food + state.wood,
    )


# ---------------------------------------------------------------------------
# BT solver: iterative Minorization-Maximization on the win matrix.
# Reference: Hunter (2004), "MM algorithms for generalized Bradley-Terry models".
# ---------------------------------------------------------------------------

_MM_TOLERANCE = 1e-6
_MM_MAX_ITERS = 1000
# Symmetric Bayesian smoothing: keeps the MLE bounded when one profile is
# undefeated (otherwise its rating diverges to +inf) and when all outcomes
# are ties (otherwise denom=0 → division crash). Adds +0.5 phantom games
# each way to every off-diagonal cell. Standard practice — same approach
# used by LMSys Chatbot Arena's BT estimator.
_BT_PRIOR = 0.5


def _solve_bt(wins: NDArray[np.int64]) -> NDArray[np.float64]:
    """Solve BT from a square integer win matrix. Returns normalised log-ratings.

    `wins[i, j]` is the number of times profile i beat profile j.
    Output sums to 0 (mean-centered) and is the log of MM-fitted strengths.

    Numpy stubs return Any from `.shape[i]`, arithmetic between arrays, and
    most ufunc calls. Operations that flow into typed downstream code are
    cast to `NDArray[float64]` / `int` at the boundary so the rest of the
    function is statically typed.
    """
    n = int(cast("int", wins.shape[0]))
    if wins.shape != (n, n):
        raise ValueError(f"wins must be square, got {wins.shape}")

    smoothed_f = cast("_F64Array", wins.astype(np.float64))
    smoothed = cast("_F64Array", smoothed_f + _BT_PRIOR * (1 - np.eye(n)))
    total_wins = cast("_F64Array", smoothed.sum(axis=1))
    games = cast("_F64Array", smoothed + smoothed.T)
    pi: NDArray[np.float64] = np.ones(n, dtype=np.float64)

    for _ in range(_MM_MAX_ITERS):
        denom: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom[i] += cast("float", games[i, j] / (pi[i] + pi[j]))
        new_pi = cast("_F64Array", total_wins / denom)
        new_pi = cast("_F64Array", new_pi / cast("float", new_pi.sum()) * n)

        if cast("float", np.max(np.abs(new_pi - pi))) < _MM_TOLERANCE:
            pi = new_pi
            break
        pi = new_pi
    else:
        raise RankingError(f"BT solver did not converge in {_MM_MAX_ITERS} iterations")

    log_ratings = cast("_F64Array", np.log(pi))
    return cast("_F64Array", log_ratings - cast("float", log_ratings.mean()))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _wins_from_outcomes(
    outcomes: list[PairwiseOutcome],
    profile_names: list[str],
) -> NDArray[np.int64]:
    """Aggregate outcomes into a win matrix over (round_idx, scenario_name) pairs."""
    n = len(profile_names)
    name_idx = {name: i for i, name in enumerate(profile_names)}
    wins = np.zeros((n, n), dtype=np.int64)

    by_match: dict[tuple[int, str], list[PairwiseOutcome]] = {}
    for o in outcomes:
        by_match.setdefault((o.round_idx, o.scenario_name), []).append(o)

    for participants in by_match.values():
        for a in participants:
            for b in participants:
                if a.profile_name == b.profile_name:
                    continue
                if a.score > b.score:
                    wins[name_idx[a.profile_name], name_idx[b.profile_name]] += 1
    return wins


def _bootstrap_ci(
    outcomes: list[PairwiseOutcome],
    profile_names: list[str],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Percentile-bootstrap 95% CIs for BT log-ratings."""
    n_outcomes = len(outcomes)
    sampled_ratings: list[NDArray[np.float64]] = []
    indices: NDArray[np.int64] = np.arange(n_outcomes)

    for _ in range(n_samples):
        sample_idx = cast("NDArray[np.int64]", rng.choice(indices, size=n_outcomes, replace=True))
        sample = [outcomes[i] for i in cast("list[int]", sample_idx.tolist())]
        try:
            wins = _wins_from_outcomes(sample, profile_names)
            sampled_ratings.append(_solve_bt(wins))
        except RankingError:
            continue  # degenerate bootstrap sample (one profile dominates)

    stacked = cast("_F64Array", np.stack(sampled_ratings, axis=0))
    lo = cast("_F64Array", np.percentile(stacked, 2.5, axis=0))
    hi = cast("_F64Array", np.percentile(stacked, 97.5, axis=0))
    ci_low = {name: float(cast("np.float64", lo[i])) for i, name in enumerate(profile_names)}
    ci_high = {name: float(cast("np.float64", hi[i])) for i, name in enumerate(profile_names)}
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# Result assembly + sink emission
# ---------------------------------------------------------------------------


def _build_result(
    outcomes: list[PairwiseOutcome],
    profile_names: list[str],
    n_samples: int,
    rng: np.random.Generator,
) -> RankingResult:
    wins = _wins_from_outcomes(outcomes, profile_names)
    point_log_ratings = _solve_bt(wins)
    ratings = {
        name: float(cast("np.float64", point_log_ratings[i]))
        for i, name in enumerate(profile_names)
    }
    ci_low, ci_high = _bootstrap_ci(outcomes, profile_names, n_samples, rng)
    pairwise_wins = {
        (a, b): int(cast("np.int64", wins[i, j]))
        for i, a in enumerate(profile_names)
        for j, b in enumerate(profile_names)
        if i != j
    }
    return RankingResult(
        ratings=ratings,
        ci_low=ci_low,
        ci_high=ci_high,
        pairwise_wins=pairwise_wins,
        n_races=len(outcomes),
    )


def _emit_ratings(sink: EventSink, ranking_id: str, result: RankingResult) -> None:
    """Persist ratings + CI bounds as MetricPayload events under a synthetic run_id."""
    ts = datetime.now(UTC)
    for profile, rating in result.ratings.items():
        for metric_name, value in (
            (f"ranking_rating_{profile}", rating),
            (f"ranking_ci_lo_{profile}", result.ci_low[profile]),
            (f"ranking_ci_hi_{profile}", result.ci_high[profile]),
        ):
            sink.emit(
                Event(
                    run_id=ranking_id,
                    agent_id=profile,
                    t=0,
                    payload=MetricPayload(name=metric_name, value=value),
                    ts=ts,
                )
            )


# ---------------------------------------------------------------------------
# Race orchestration
# ---------------------------------------------------------------------------


def _select_scenarios(config: RankingConfig) -> list[Scenario]:
    if not config.scenarios:
        return list(DEFAULT_SCENARIOS)
    return [get_scenario(name) for name in config.scenarios]


async def _collect_outcomes(
    config: RankingConfig,
    scenarios: list[Scenario],
    race_fn: RaceFn,
    sink: EventSink,
    score_fn: ScoreFn,
) -> list[PairwiseOutcome]:
    race_config = RaceConfig(turns=config.turns, profiles=list(config.profiles))
    outcomes: list[PairwiseOutcome] = []
    for round_idx in range(config.rounds):
        for scenario in scenarios:
            variant_results = await race_fn(race_config, scenario.initial_state, sink)
            for vr in variant_results:
                final_state = vr.loop_result.turns[-1].state_after
                outcomes.append(
                    PairwiseOutcome(
                        profile_name=vr.profile.name,
                        scenario_name=scenario.name,
                        round_idx=round_idx,
                        score=score_fn(final_state),
                    )
                )
    return outcomes


async def _rank_with_race_fn(
    config: RankingConfig,
    race_fn: RaceFn,
    sink: EventSink,
    score_fn: ScoreFn,
    ranking_id: str,
) -> RankingResult:
    scenarios = _select_scenarios(config)
    outcomes = await _collect_outcomes(config, scenarios, race_fn, sink, score_fn)
    rng = np.random.default_rng(config.bootstrap_seed)
    profile_names = [p.name for p in config.profiles]
    result = _build_result(outcomes, profile_names, config.bootstrap_samples, rng)
    _emit_ratings(sink, ranking_id, result)
    return result


async def rank(
    config: RankingConfig,
    sink: EventSink | None = None,
    score_fn: ScoreFn = composite_score,
    ranking_id: str = "ranking",
) -> RankingResult:
    """Run K rounds x M scenarios x N profiles against the real API; rank them."""
    effective_sink: EventSink = sink if sink is not None else NullEventSink()

    async def real_race(cfg: RaceConfig, state: WorldState, s: EventSink) -> list[VariantResult]:
        return await race(cfg, state, s)

    return await _rank_with_race_fn(config, real_race, effective_sink, score_fn, ranking_id)


async def rank_with_mock(
    config: RankingConfig,
    sink: EventSink | None = None,
    score_fn: ScoreFn = composite_score,
    ranking_id: str = "ranking",
) -> RankingResult:
    """Offline ranking using the deterministic mock invoke. No API key needed."""
    effective_sink: EventSink = sink if sink is not None else NullEventSink()

    async def mock_race(cfg: RaceConfig, state: WorldState, s: EventSink) -> list[VariantResult]:
        return await race_with_mock(cfg, state, s)

    return await _rank_with_race_fn(config, mock_race, effective_sink, score_fn, ranking_id)


# Re-export for test direct access.
_solve_bt_for_test = _solve_bt
_bootstrap_ci_for_test = _bootstrap_ci
_wins_from_outcomes_for_test = _wins_from_outcomes
