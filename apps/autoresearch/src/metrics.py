"""Composite scoring — turns an AgentMemory metrics snapshot into one 0-1 score."""

from __future__ import annotations

from collections.abc import Mapping  # runtime use: the `Metrics` alias below
from dataclasses import dataclass

# Normalization caps — scores saturate at these values
MAX_SURVIVAL_SECONDS = 1200.0  # 20 minutes
MAX_FOOD_GATHERED = 5000
# A human Dark-to-Feudal is about this; faster than the reference scores 1.0.
REFERENCE_AGE_UP_SECONDS = 600.0

# Weights must sum to 1.0. Version 2 (plan 2.3) puts age progress and age SPEED
# in charge: under v1 (survival 0.30, population 0.25, age 0.20) the only game
# that ever reached Feudal ranked 3rd of 14, behind two that never aged up.
WEIGHT_AGE = 0.40
WEIGHT_AGE_SPEED = 0.25
WEIGHT_ECONOMY = 0.20
WEIGHT_ACTION_SUCCESS = 0.10
WEIGHT_SURVIVAL = 0.05

AGE_SCORES = {
    "Dark Age": 0.0,
    "Feudal Age": 0.33,
    "Castle Age": 0.66,
    "Imperial Age": 1.0,
}

# The snapshot crosses an untyped boundary (OCR, LLM observations), so values
# are read as `object` and narrowed rather than trusted.
Metrics = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GameScore:
    """A game's composite score and its components."""

    composite: float
    age: float
    age_speed: float
    economy: float
    action_success: float
    survival: float
    raw_metrics: Metrics


def _number(metrics: Metrics, key: str, default: float = 0.0) -> float:
    """A numeric field, or `default` when absent or non-numeric."""
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _age_speed(metrics: Metrics) -> float:
    """How fast the agent left the Dark Age; 0.0 when it never did."""
    seconds = _number(metrics, "feudal_time_s")
    return min(1.0, REFERENCE_AGE_UP_SECONDS / seconds) if seconds > 0 else 0.0


def _age_progress(metrics: Metrics) -> float:
    """Age reached, counting a victory as Imperial.

    A victory must never set the composite to 1.0: it arrives on
    `observations.game_state`, the channel `memory.update_age` refuses to trust
    for age, so one hallucination would top a tournament permanently.
    """
    if metrics.get("game_end_reason") == "victory":
        return 1.0
    return _number(metrics, "age_score")


def compute_score(metrics: Metrics) -> GameScore:
    """Score one game from `AgentMemory.get_metrics_snapshot()`."""
    age = _age_progress(metrics)
    age_speed = _age_speed(metrics)
    economy = min(_number(metrics, "total_food_gathered") / MAX_FOOD_GATHERED, 1.0)
    action_success = _number(metrics, "action_success_rate")
    survival = min(_number(metrics, "survival_time") / MAX_SURVIVAL_SECONDS, 1.0)

    composite = (
        WEIGHT_AGE * age
        + WEIGHT_AGE_SPEED * age_speed
        + WEIGHT_ECONOMY * economy
        + WEIGHT_ACTION_SUCCESS * action_success
        + WEIGHT_SURVIVAL * survival
    )

    return GameScore(
        composite=round(composite, 4),
        age=round(age, 4),
        age_speed=round(age_speed, 4),
        economy=round(economy, 4),
        action_success=round(action_success, 4),
        survival=round(survival, 4),
        raw_metrics=metrics,
    )
