"""The single state view every rule is evaluated against."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..turn_timing import elapsed_ms

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core import WorldState

    from ..memory import GameState

_NO_JOBS: Mapping[str, int] = MappingProxyType({})


def _no_jobs() -> Mapping[str, int]:
    """Shared empty view. A factory because `mappingproxy` is an illegal
    dataclass default on Python 3.11, where it is still unhashable."""
    return _NO_JOBS


# Villagers the game starts with (mirrors memory.INITIAL_POPULATION).
_STARTING_VILLAGERS = 4


@dataclass(frozen=True, slots=True)
class PolicyState:
    """Everything a rule may read, from either the real game or the simulator.

    Genuinely immutable: the actor shares one snapshot across policy paths, so
    `villager_jobs` is a read-only view, not a dict a caller could mutate.
    """

    age: str = "Dark Age"
    age_known: bool = True
    food: int = 0
    wood: int = 0
    gold: int = 0
    stone: int = 0
    population: int = 0
    population_cap: int = 0
    population_known: bool = True
    villagers_ordered: int = _STARTING_VILLAGERS
    buildings_seen: frozenset[str] = frozenset()
    pending_buildings: frozenset[str] = frozenset()
    building_purchases: frozenset[str] = frozenset()
    pending_research: frozenset[str] = frozenset()
    age_up_paid: frozenset[str] = frozenset()
    research_purchases: frozenset[str] = frozenset()
    researched: frozenset[str] = frozenset()
    pending_actions: frozenset[str] = frozenset()
    suppressed_actions: frozenset[str] = frozenset()
    reserved_resources: Mapping[str, int] = field(default_factory=_no_jobs)
    pending_population: int = 0
    visible_classes: frozenset[str] = frozenset()
    own_army_present: bool = False
    spatial_valid: bool = True
    known_resources: frozenset[str] = frozenset({"food", "wood", "gold", "stone"})
    idle_present: bool | None = None
    idle_count: int | None = None
    idle_streak: int = 0
    villager_jobs: Mapping[str, int] = field(default_factory=_no_jobs)
    turn: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        object.__setattr__(self, "villager_jobs", MappingProxyType(dict(self.villager_jobs)))
        object.__setattr__(
            self, "reserved_resources", MappingProxyType(dict(self.reserved_resources))
        )

    @property
    def age_ms(self) -> float:
        """Milliseconds since this snapshot was taken."""
        return elapsed_ms(self.captured_at)


def _as_int(value: object) -> int:
    """Coerce a resource reading; OCR and LLM values cross an untyped boundary."""
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def from_game_state(
    state: GameState,
    *,
    captured_at: float,
    villager_jobs: Mapping[str, int] | None = None,
    pending_buildings: frozenset[str] = frozenset(),
    known_resources: frozenset[str] | None = None,
) -> PolicyState:
    """Build from the real game's `memory.GameState`.

    `captured_at` is when the frame was grabbed, not when this call runs.
    Required: letting it default made every `max_state_age_ms` a tautology.
    """
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
        pending_buildings=pending_buildings,
        known_resources=(
            known_resources
            if known_resources is not None
            else frozenset({"food", "wood", "gold", "stone"})
        ),
        idle_present=state.idle_present,
        idle_count=state.idle_count,
        idle_streak=state.idle_streak,
        villager_jobs=jobs,
        captured_at=captured_at,
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
