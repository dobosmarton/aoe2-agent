"""The single state view every rule is evaluated against."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core import WorldState

    from ..memory import GameState

_NO_JOBS: Mapping[str, int] = MappingProxyType({})

# Villagers the game starts with (mirrors memory.INITIAL_POPULATION).
_STARTING_VILLAGERS = 4


@dataclass(frozen=True, slots=True)
class PolicyState:
    """Everything a rule may read, from either the real game or the simulator.

    Genuinely immutable: Phase 3 shares one snapshot across three loops, so
    `villager_jobs` is a read-only view, not a dict a caller could mutate.
    """

    age: str = "Dark Age"
    food: int = 0
    wood: int = 0
    gold: int = 0
    stone: int = 0
    population: int = 0
    population_cap: int = 0
    villagers_ordered: int = _STARTING_VILLAGERS
    buildings_seen: frozenset[str] = frozenset()
    idle_present: bool | None = None
    idle_count: int | None = None
    idle_streak: int = 0
    villager_jobs: Mapping[str, int] = _NO_JOBS
    turn: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def age_ms(self) -> float:
        """Milliseconds since this snapshot was taken."""
        return (time.monotonic() - self.captured_at) * 1000.0


def _as_int(value: object) -> int:
    """Coerce a resource reading; OCR and LLM values cross an untyped boundary."""
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def from_game_state(
    state: GameState, *, villager_jobs: Mapping[str, int] | None = None
) -> PolicyState:
    """Build from the real game's `memory.GameState`."""
    resources = state.resources
    jobs = _NO_JOBS if villager_jobs is None else MappingProxyType(dict(villager_jobs))
    return PolicyState(
        age=state.current_age,
        food=_as_int(resources.get("food", 0)),
        wood=_as_int(resources.get("wood", 0)),
        gold=_as_int(resources.get("gold", 0)),
        stone=_as_int(resources.get("stone", 0)),
        population=state.population,
        population_cap=state.population_cap,
        villagers_ordered=state.villagers_ordered,
        buildings_seen=state.buildings_seen,
        idle_present=state.idle_present,
        idle_count=state.idle_count,
        idle_streak=state.idle_streak,
        villager_jobs=jobs,
    )


def from_world_state(state: WorldState) -> PolicyState:
    """Build from the simulator's `core.WorldState`.

    The simulator has no idle badge, so idle dispatch stays dormant there until
    Phase 5.2 renders resources.
    """
    return PolicyState(
        age=state.age,
        food=int(state.food),
        wood=int(state.wood),
        gold=int(state.gold),
        stone=int(state.stone),
        population=state.population,
        population_cap=state.pop_cap,
        villagers_ordered=state.population + len(state.villager_queue),
        buildings_seen=frozenset(state.buildings),
        turn=state.turn,
    )
