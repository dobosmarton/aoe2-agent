"""Reviewed economic actions that a policy may choose between.

Candidates express affordances, not strategy. Code decides whether an action is
possible and safe; a policy decides which currently possible action is useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .state import PolicyState

CandidateId = Literal[
    "wait",
    "advance_to_feudal",
    "build_house",
    "queue_villager",
    "build_mill",
    "build_lumber_camp",
    "build_mining_camp",
]

_HOUSE_HEADROOM_LIMIT = 4
_POPULATION_CAP_LIMIT = 200
_VILLAGER_FOOD_COST = 50
_HOUSE_WOOD_COST = 25
_ECONOMY_BUILDING_WOOD_COST = 100
_FEUDAL_FOOD_COST = 500


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """One bounded policy option backed by reviewed action templates."""

    id: CandidateId
    description: str
    cost: Mapping[str, int]
    actions: tuple[Mapping[str, object], ...]

    def render(self) -> list[dict[str, object]]:
        """Return fresh dictionaries for the validation and execution boundary."""
        return [dict(action) for action in self.actions]


def feasible_candidates(state: PolicyState) -> tuple[ActionCandidate, ...]:
    """Return every economic action the current state can safely execute.

    The checks here are game facts: costs, prerequisites, caps and duplicate
    buildings. Strategic timing deliberately does not belong here.
    """
    candidates = [_wait_candidate()]

    if _can_advance_to_feudal(state):
        candidates.append(_advance_to_feudal_candidate())
    if _needs_house_and_can_build(state):
        candidates.append(_build_candidate("build_house", "q", "house", _HOUSE_WOOD_COST))
    if _can_queue_villager(state):
        candidates.append(_queue_villager_candidate())
    if _can_build_unique(state, "mill"):
        candidates.append(_build_candidate("build_mill", "w", "mill"))
    if _can_build_unique(state, "lumber_camp"):
        candidates.append(_build_candidate("build_lumber_camp", "r", "lumber camp"))
    if _can_build_unique(state, "mining_camp"):
        candidates.append(_build_candidate("build_mining_camp", "e", "mining camp"))

    return tuple(candidates)


def find_candidate(
    candidates: tuple[ActionCandidate, ...], candidate_id: str
) -> ActionCandidate | None:
    """Find a selected candidate without trusting an external string."""
    return next((candidate for candidate in candidates if candidate.id == candidate_id), None)


def _can_advance_to_feudal(state: PolicyState) -> bool:
    prerequisites = {"mill", "lumber_camp"}
    return (
        state.age == "Dark Age"
        and state.food >= _FEUDAL_FOOD_COST
        and prerequisites.issubset(state.buildings_seen)
    )


def _needs_house_and_can_build(state: PolicyState) -> bool:
    headroom = state.population_cap - state.population
    return (
        0 < state.population_cap < _POPULATION_CAP_LIMIT
        and headroom <= _HOUSE_HEADROOM_LIMIT
        and state.wood >= _HOUSE_WOOD_COST
    )


def _can_queue_villager(state: PolicyState) -> bool:
    return (
        state.population_cap > 0
        and state.villagers_ordered < state.population_cap
        and state.food >= _VILLAGER_FOOD_COST
    )


def _can_build_unique(state: PolicyState, building: str) -> bool:
    return building not in state.buildings_seen and state.wood >= _ECONOMY_BUILDING_WOOD_COST


def _wait_candidate() -> ActionCandidate:
    return _candidate(
        candidate_id="wait",
        description=(
            "Spend nothing now. Preserve resources for a more important purchase while the "
            "economy continues gathering."
        ),
    )


def _advance_to_feudal_candidate() -> ActionCandidate:
    return _candidate(
        candidate_id="advance_to_feudal",
        description=(
            "Research the Feudal Age now. The required food and prerequisite buildings are "
            "already available."
        ),
        cost={"food": _FEUDAL_FOOD_COST},
        actions=(
            {"type": "press", "key": "h", "intent": "Select TC for Feudal research"},
            {"type": "press", "key": "z", "intent": "Research Feudal Age (TypeSafe)"},
        ),
    )


def _queue_villager_candidate() -> ActionCandidate:
    return _candidate(
        candidate_id="queue_villager",
        description=(
            "Queue one villager to grow the economy. Food and population capacity are available."
        ),
        cost={"food": _VILLAGER_FOOD_COST},
        actions=({"type": "queue_villager", "intent": "Queue villager (TypeSafe)"},),
    )


def _build_candidate(
    candidate_id: CandidateId,
    building_key: str,
    building_name: str,
    wood_cost: int = _ECONOMY_BUILDING_WOOD_COST,
) -> ActionCandidate:
    descriptions = {
        "house": "Add population capacity before production becomes blocked.",
        "mill": "Unlock farms and satisfy one Dark Age prerequisite for Feudal Age.",
        "lumber camp": "Improve wood income and satisfy one Dark Age prerequisite for Feudal Age.",
        "mining camp": "Enable efficient gold or stone gathering for later-age requirements.",
    }
    return _candidate(
        candidate_id=candidate_id,
        description=descriptions[building_name],
        cost={"wood": wood_cost},
        actions=(
            {
                "type": "build",
                "building_key": building_key,
                "intent": f"Build {building_name} (TypeSafe)",
            },
        ),
    )


def _candidate(
    candidate_id: CandidateId,
    description: str,
    cost: dict[str, int] | None = None,
    actions: tuple[dict[str, object], ...] = (),
) -> ActionCandidate:
    readonly_cost: Mapping[str, int] = MappingProxyType(dict(cost or {}))
    readonly_actions = tuple(MappingProxyType(dict(action)) for action in actions)
    return ActionCandidate(
        id=candidate_id,
        description=description,
        cost=readonly_cost,
        actions=readonly_actions,
    )


__all__ = ["ActionCandidate", "CandidateId", "feasible_candidates", "find_candidate"]
