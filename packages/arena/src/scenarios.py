"""Named starting `WorldState` scenarios for the synth arena (Phase 8).

A scenario pairs a stable identifier with an initial WorldState so the
ranking harness can run profiles across diverse game-state starts.
`DEFAULT_SCENARIOS` covers four common openings; callers needing more
variety should add entries here rather than constructing WorldStates ad-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.world_sim import WorldState


@dataclass(frozen=True, slots=True)
class Scenario:
    """One named starting position for a race."""

    name: str
    initial_state: WorldState


def _balanced() -> WorldState:
    return WorldState(
        food=200.0,
        wood=150.0,
        gold=0.0,
        stone=0.0,
        population=8,
        pop_cap=25,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def _food_poor() -> WorldState:
    return WorldState(
        food=80.0,
        wood=200.0,
        gold=0.0,
        stone=0.0,
        population=8,
        pop_cap=25,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def _wood_poor() -> WorldState:
    return WorldState(
        food=300.0,
        wood=40.0,
        gold=0.0,
        stone=0.0,
        population=8,
        pop_cap=25,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def _late_start() -> WorldState:
    return WorldState(
        food=120.0,
        wood=120.0,
        gold=0.0,
        stone=0.0,
        population=6,
        pop_cap=20,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="balanced", initial_state=_balanced()),
    Scenario(name="food-poor", initial_state=_food_poor()),
    Scenario(name="wood-poor", initial_state=_wood_poor()),
    Scenario(name="late-start", initial_state=_late_start()),
)


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name. Raises KeyError on unknown name."""
    for scenario in DEFAULT_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)
