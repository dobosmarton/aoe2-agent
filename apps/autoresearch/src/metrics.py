"""Composite scoring for autoresearch experiments.

Converts raw game metrics from AgentMemory into a single 0-1 score
that can be compared across experiments (like val_bpb in autoresearch).
"""

from dataclasses import dataclass

# Normalization caps — scores saturate at these values
MAX_SURVIVAL_SECONDS = 1200.0  # 20 minutes
MAX_FOOD_GATHERED = 5000
# A human Dark-to-Feudal is about this; faster than the reference scores 1.0.
REFERENCE_AGE_UP_SECONDS = 600.0

# Score weights — must sum to 1.0. Version 2 (plan 2.3): age progress and age
# SPEED carry the score. Under version 1 (survival 0.30, population 0.25, age
# 0.20) the only game that ever reached Feudal ranked 3rd of 14, behind two
# games that never left the Dark Age.
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


@dataclass
class GameScore:
    """Breakdown of a game's composite score."""

    composite: float
    age: float
    age_speed: float
    economy: float
    action_success: float
    survival: float
    raw_metrics: dict


def _age_speed(metrics: dict) -> float:
    """How fast the agent left the Dark Age; 0.0 when it never did."""
    seconds = metrics.get("feudal_time_s")
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return 0.0
    return min(1.0, REFERENCE_AGE_UP_SECONDS / seconds)


def _age_progress(metrics: dict) -> float:
    """Age reached, with a victory scored as Imperial.

    A victory is NOT allowed to set the composite to 1.0. It arrives via
    `observations.game_state`, the same executor channel `memory.update_age`
    refuses to trust for age, so one hallucination would top a tournament
    permanently.
    """
    if metrics.get("game_end_reason") == "victory":
        return 1.0
    return metrics.get("age_score", 0.0)


def compute_score(metrics: dict) -> GameScore:
    """Compute composite game score from AgentMemory metrics snapshot.

    Args:
        metrics: Output of AgentMemory.get_metrics_snapshot()

    Returns:
        GameScore with composite score and per-component breakdown.
    """
    age = _age_progress(metrics)
    age_speed = _age_speed(metrics)
    economy = min(metrics.get("total_food_gathered", 0) / MAX_FOOD_GATHERED, 1.0)
    action_success = metrics.get("action_success_rate", 0.0)
    survival = min(metrics.get("survival_time", 0) / MAX_SURVIVAL_SECONDS, 1.0)

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
