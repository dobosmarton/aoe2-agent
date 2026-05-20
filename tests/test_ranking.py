"""Tests for arena/ranking.py (Phase 8). Offline."""

from __future__ import annotations

import asyncio

import numpy as np

from arena.config_profile import ConfigProfile, RankingConfig
from arena.ranking import (
    PairwiseOutcome,
    RankingResult,
    _bootstrap_ci_for_test,
    _solve_bt_for_test,
    _wins_from_outcomes_for_test,
    composite_score,
    rank_with_mock,
)
from evaluation.event_log import Event, MetricPayload
from evaluation.world_sim import WorldState


def _state(food: float = 0.0, wood: float = 0.0, pop: int = 0, age: str = "Dark Age") -> WorldState:
    return WorldState(
        food=food,
        wood=wood,
        gold=0.0,
        stone=0.0,
        population=pop,
        pop_cap=25,
        age=age,
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------


def test_composite_score_orders_feudal_above_dark() -> None:
    assert composite_score(_state(age="Feudal Age")) > composite_score(_state(age="Dark Age"))


def test_composite_score_uses_pop_to_break_age_tie() -> None:
    assert composite_score(_state(pop=22)) > composite_score(_state(pop=18))


def test_composite_score_uses_resources_to_break_pop_tie() -> None:
    a = composite_score(_state(food=500, wood=100, pop=22))
    b = composite_score(_state(food=200, wood=200, pop=22))
    assert a > b


# ---------------------------------------------------------------------------
# _solve_bt — point estimates
# ---------------------------------------------------------------------------


def test_solve_bt_recovers_ratings_for_two_profile_case() -> None:
    wins = np.array([[0, 9], [1, 0]], dtype=np.int64)  # A beats B 9-1
    log_ratings = _solve_bt_for_test(wins)
    assert log_ratings[0] > log_ratings[1]


def test_solve_bt_three_profile_transitivity() -> None:
    # A beats B, B beats C, A beats C → A > B > C
    wins = np.array(
        [
            [0, 8, 8],
            [2, 0, 7],
            [2, 3, 0],
        ],
        dtype=np.int64,
    )
    log_ratings = _solve_bt_for_test(wins)
    assert log_ratings[0] > log_ratings[1] > log_ratings[2]


# ---------------------------------------------------------------------------
# _solve_bt — invariant: normalised log-ratings sum to 0
# ---------------------------------------------------------------------------


def test_solve_bt_ratings_sum_to_zero_two_profile() -> None:
    wins = np.array([[0, 9], [1, 0]], dtype=np.int64)
    assert abs(_solve_bt_for_test(wins).sum()) < 1e-9


def test_solve_bt_ratings_sum_to_zero_three_profile() -> None:
    wins = np.array([[0, 8, 5], [2, 0, 7], [5, 3, 0]], dtype=np.int64)
    assert abs(_solve_bt_for_test(wins).sum()) < 1e-9


def test_solve_bt_ratings_sum_to_zero_skewed_matrix() -> None:
    # One dominant winner — solver should still produce mean-centered output.
    wins = np.array([[0, 20, 20], [0, 0, 5], [0, 5, 0]], dtype=np.int64)
    assert abs(_solve_bt_for_test(wins).sum()) < 1e-9


# ---------------------------------------------------------------------------
# _bootstrap_ci
# ---------------------------------------------------------------------------


def _two_profile_outcomes() -> list[PairwiseOutcome]:
    # A beats B 8 out of 10 rounds.
    outcomes: list[PairwiseOutcome] = []
    for r in range(10):
        a_score = (1, 22, 500.0) if r < 8 else (0, 18, 100.0)
        b_score = (0, 18, 100.0) if r < 8 else (1, 22, 500.0)
        outcomes.append(PairwiseOutcome("A", "s", r, a_score))
        outcomes.append(PairwiseOutcome("B", "s", r, b_score))
    return outcomes


def test_bootstrap_ci_lower_le_point_estimate() -> None:
    outcomes = _two_profile_outcomes()
    rng = np.random.default_rng(0)
    wins = _wins_from_outcomes_for_test(outcomes, ["A", "B"])
    point = _solve_bt_for_test(wins)
    ci_low, _ = _bootstrap_ci_for_test(outcomes, ["A", "B"], 200, rng)
    assert ci_low["A"] <= point[0] + 1e-9


def test_bootstrap_ci_upper_ge_point_estimate() -> None:
    outcomes = _two_profile_outcomes()
    rng = np.random.default_rng(0)
    wins = _wins_from_outcomes_for_test(outcomes, ["A", "B"])
    point = _solve_bt_for_test(wins)
    _, ci_high = _bootstrap_ci_for_test(outcomes, ["A", "B"], 200, rng)
    assert ci_high["A"] >= point[0] - 1e-9


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    outcomes = _two_profile_outcomes()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    ci_low_1, _ = _bootstrap_ci_for_test(outcomes, ["A", "B"], 200, rng1)
    ci_low_2, _ = _bootstrap_ci_for_test(outcomes, ["A", "B"], 200, rng2)
    assert ci_low_1 == ci_low_2


# ---------------------------------------------------------------------------
# rank_with_mock — integration
# ---------------------------------------------------------------------------


def _ranking_config() -> RankingConfig:
    return RankingConfig(
        turns=3,
        profiles=[
            ConfigProfile(name="profile-a"),
            ConfigProfile(name="profile-b"),
        ],
        scenarios=["balanced"],
        rounds=2,
        bootstrap_samples=100,
    )


def test_rank_with_mock_returns_one_rating_per_profile() -> None:
    result = asyncio.run(rank_with_mock(_ranking_config()))
    assert set(result.ratings) == {"profile-a", "profile-b"}


def test_rank_with_mock_emits_metric_payload_per_profile() -> None:
    sink = _RecordingSink()
    asyncio.run(rank_with_mock(_ranking_config(), sink=sink))
    rating_events = [
        e
        for e in sink.events
        if isinstance(e.payload, MetricPayload) and e.payload.name.startswith("ranking_rating_")
    ]
    assert len(rating_events) == 2


def test_rank_with_mock_returns_ranking_result_type() -> None:
    result = asyncio.run(rank_with_mock(_ranking_config()))
    assert isinstance(result, RankingResult)
