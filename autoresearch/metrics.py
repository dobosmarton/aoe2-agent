"""Composite scoring for autoresearch experiments.

Converts raw game metrics from AgentMemory into a single 0-1 score
that can be compared across experiments (like val_bpb in autoresearch).
"""

from dataclasses import dataclass

# Normalization caps — scores saturate at these values
MAX_SURVIVAL_SECONDS = 1200.0  # 20 minutes
MAX_POPULATION = 50
MAX_FOOD_GATHERED = 5000

# Score weights — must sum to 1.0
WEIGHT_SURVIVAL = 0.30
WEIGHT_POPULATION = 0.25
WEIGHT_AGE = 0.20
WEIGHT_ECONOMY = 0.15
WEIGHT_ACTION_SUCCESS = 0.10

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
    survival: float
    population: float
    age: float
    economy: float
    action_success: float
    raw_metrics: dict


def compute_score(metrics: dict) -> GameScore:
    """Compute composite game score from AgentMemory metrics snapshot.

    Args:
        metrics: Output of AgentMemory.get_metrics_snapshot()

    Returns:
        GameScore with composite score and per-component breakdown.
    """
    survival = min(metrics.get("survival_time", 0) / MAX_SURVIVAL_SECONDS, 1.0)
    population = min(metrics.get("peak_population", 0) / MAX_POPULATION, 1.0)
    age = metrics.get("age_score", 0.0)
    economy = min(metrics.get("total_food_gathered", 0) / MAX_FOOD_GATHERED, 1.0)
    action_success = metrics.get("action_success_rate", 0.0)

    composite = (
        WEIGHT_SURVIVAL * survival
        + WEIGHT_POPULATION * population
        + WEIGHT_AGE * age
        + WEIGHT_ECONOMY * economy
        + WEIGHT_ACTION_SUCCESS * action_success
    )

    return GameScore(
        composite=round(composite, 4),
        survival=round(survival, 4),
        population=round(population, 4),
        age=round(age, 4),
        economy=round(economy, 4),
        action_success=round(action_success, 4),
        raw_metrics=metrics,
    )
