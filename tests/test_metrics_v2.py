"""The v2 objective, checked against the committed experiment ledger.

Under v1 the only game that ever reached Feudal ranked 3rd of 14, behind two
that never aged up. These tests pin the fix.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest
from autoresearch import metrics
from autoresearch.metrics import GameScore, compute_score
from gameplay_agent.memory import AgentMemory, Turn

LEDGER = Path(__file__).parent.parent / "experiments" / "results.tsv"

# The only ledger row that ever reached the Feudal Age, and the Dark Age row
# that outscored it under v1.
_FEUDAL_ROW = "exp_0013"
_DARK_AGE_WINNER_V1 = "exp_0012"


def _ledger_rows() -> list[dict[str, str]]:
    with LEDGER.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _rescore_v1_row(row: dict[str, str]) -> float:
    """Score a v1 row under v2. Its age-up time was never recorded, so the
    speed term is 0 — honest, and it still flips the ranking."""
    return (
        metrics.WEIGHT_AGE * float(row["age"])
        + metrics.WEIGHT_AGE_SPEED * 0.0
        + metrics.WEIGHT_ECONOMY * float(row["economy"])
        + metrics.WEIGHT_ACTION_SUCCESS * float(row["action_success"])
        + metrics.WEIGHT_SURVIVAL * float(row["survival"])
    )


# ---------------------------------------------------------------------------
# The inversion, on real recorded games
# ---------------------------------------------------------------------------


def test_only_one_recorded_game_ever_left_the_dark_age() -> None:
    """If this changes, the fixture below needs revisiting."""
    aged_up = [r["experiment_id"] for r in _ledger_rows() if float(r["age"]) > 0]
    assert aged_up == [_FEUDAL_ROW]


def test_v1_ranked_a_dark_age_game_above_the_only_feudal_game() -> None:
    """The bug this phase fixes, still visible in the committed scores."""
    rows = {r["experiment_id"]: float(r["composite_score"]) for r in _ledger_rows()}
    assert rows[_DARK_AGE_WINNER_V1] > rows[_FEUDAL_ROW]


def test_v2_ranks_the_only_feudal_game_first() -> None:
    ranked = sorted(_ledger_rows(), key=_rescore_v1_row, reverse=True)
    assert ranked[0]["experiment_id"] == _FEUDAL_ROW


# ---------------------------------------------------------------------------
# Weights and components
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    total = (
        metrics.WEIGHT_AGE
        + metrics.WEIGHT_AGE_SPEED
        + metrics.WEIGHT_ECONOMY
        + metrics.WEIGHT_ACTION_SUCCESS
        + metrics.WEIGHT_SURVIVAL
    )
    assert total == pytest.approx(1.0)


def test_age_outweighs_survival() -> None:
    """The whole point of v2: reaching an age beats sitting still."""
    assert metrics.WEIGHT_AGE > metrics.WEIGHT_SURVIVAL * 4


@pytest.mark.parametrize(
    ("feudal_time_s", "expected"),
    [
        (None, 0.0),  # never aged up
        (0, 0.0),  # guard against divide-by-zero
        (600, 1.0),  # exactly the reference
        (300, 1.0),  # faster than reference saturates
        (1200, 0.5),  # twice the reference
    ],
    ids=["never", "zero", "reference", "faster", "slower"],
)
def test_age_speed_scoring(feudal_time_s: float | None, expected: float) -> None:
    score = compute_score({"age_score": 0.33, "feudal_time_s": feudal_time_s})
    assert score.age_speed == pytest.approx(expected)


def test_population_is_no_longer_scored() -> None:
    """A huge population must not move the composite on its own."""
    base = compute_score({"age_score": 0.0})
    with_pop = compute_score({"age_score": 0.0, "peak_population": 200})
    assert base.composite == with_pop.composite


# ---------------------------------------------------------------------------
# Victory is recorded, but never overrides
# ---------------------------------------------------------------------------


def test_victory_scores_the_age_component_at_imperial() -> None:
    score = compute_score({"age_score": 0.33, "game_end_reason": "victory"})
    assert score.age == 1.0


def test_victory_does_not_set_the_composite_to_one() -> None:
    """`game_state` is executor-self-reported — the same channel `update_age`
    refuses to trust. One hallucination must not top a tournament forever."""
    score = compute_score({"age_score": 0.0, "game_end_reason": "victory"})
    assert score.composite < 1.0


def test_a_real_win_still_outranks_a_perfect_dark_age_game() -> None:
    win = compute_score({"age_score": 0.0, "game_end_reason": "victory"})
    perfect_dark_age = compute_score(
        {
            "age_score": 0.0,
            "total_food_gathered": 999_999,
            "action_success_rate": 1.0,
            "survival_time": 999_999,
        }
    )
    assert win.composite > perfect_dark_age.composite


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


def test_pareto_axes_match_the_score_components() -> None:
    from autoresearch.pareto import AXES

    fields = {f for f in GameScore.__dataclass_fields__ if f not in ("composite", "raw_metrics")}
    assert set(AXES) == fields


def test_pareto_no_longer_ranks_on_population() -> None:
    from autoresearch.pareto import AXES

    assert "population" not in AXES


# ---------------------------------------------------------------------------
# The age clock (plan 2.1)
# ---------------------------------------------------------------------------


def _started_game() -> AgentMemory:
    """A memory whose clock is running — `add_turn` starts it."""
    memory = AgentMemory()
    memory.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[]))
    return memory


def test_reaching_an_age_records_the_time() -> None:
    memory = _started_game()
    memory.update_age("Feudal Age")
    assert memory.get_metrics_snapshot()["feudal_time_s"] is not None


def test_an_age_never_reached_has_no_time() -> None:
    assert _started_game().get_metrics_snapshot()["feudal_time_s"] is None


def test_only_the_first_arrival_at_an_age_is_stamped() -> None:
    """The strategist re-reports the same age every turn."""
    memory = _started_game()
    memory.update_age("Feudal Age")
    first = memory.age_times["Feudal Age"]
    time.sleep(0.01)
    memory.update_age("Feudal Age")
    assert memory.age_times["Feudal Age"] == first
