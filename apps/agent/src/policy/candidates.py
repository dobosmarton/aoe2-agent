"""Pure eligibility and executable descriptions for the bounded action catalog."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from ..entity_utils import GATHER_CLASSES_BY_KIND, RESOURCE_KINDS, ResourceKind
from .catalog import (
    AGE_ORDER,
    BY_ID,
    CASTLE_BUILDINGS,
    DARK_BUILDINGS,
    FEUDAL_BUILDINGS,
    SPECS,
    ActionSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .state import PolicyState

CandidateId: TypeAlias = str


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """One eligible action; templates are copied at the validation boundary."""

    id: CandidateId
    description: str
    cost: Mapping[str, int]
    actions: tuple[Mapping[str, object], ...]

    def render(self) -> list[dict[str, object]]:
        return [dict(action) for action in self.actions]


def eligible(spec: ActionSpec, state: PolicyState) -> bool:
    """Game feasibility only. Goals and preferred timing belong to the selector."""
    if spec.id == "wait":
        return True
    if spec.id in state.pending_actions or spec.id in state.suppressed_actions:
        return False
    if not state.age_known and (spec.kind == "research" or spec.age != "Dark Age"):
        return False
    if state.age not in AGE_ORDER or AGE_ORDER.index(state.age) < AGE_ORDER.index(spec.age):
        return False
    if not spec.requires.issubset(state.buildings_seen):
        return False
    if any(
        kind not in state.known_resources
        or getattr(state, kind) - state.reserved_resources.get(kind, 0) < price
        for kind, price in spec.cost
    ):
        return False

    if spec.kind == "build":
        if not state.spatial_valid or spec.subject in state.pending_buildings:
            return False
        if state.idle_present is not True and "villager" not in state.visible_classes:
            return False
        if spec.unique and spec.subject in state.buildings_seen:
            return False
        if spec.subject == "house" and (
            not state.population_known
            or not 0 < state.population_cap < 200
            or state.population_cap - state.population - state.pending_population > 4
        ):
            return False
        if spec.subject == "lumber_camp" and "tree" not in state.visible_classes:
            return False
        if (
            spec.subject == "mining_camp"
            and not {"gold_mine", "stone_mine"} & state.visible_classes
        ):
            return False
    if spec.kind == "research":
        if (
            spec.subject in state.researched
            or spec.subject in state.pending_research
            or spec.subject in state.research_purchases
        ):
            return False
        if spec.subject == "feudal_age" and (
            state.age != "Dark Age" or len(state.buildings_seen & DARK_BUILDINGS) < 2
        ):
            return False
        if spec.subject == "castle_age" and (
            state.age != "Feudal Age" or len(state.buildings_seen & FEUDAL_BUILDINGS) < 2
        ):
            return False
        if spec.subject == "imperial_age" and (
            state.age != "Castle Age" or len(state.buildings_seen & CASTLE_BUILDINGS) < 2
        ):
            return False
    if spec.kind == "train" and (
        not state.population_known
        or state.population_cap <= 0
        or state.population + state.pending_population >= state.population_cap
    ):
        return False
    if spec.kind == "assign":
        if not state.spatial_valid or not state.idle_present or state.assignment_pending:
            return False
        if spec.subject not in RESOURCE_KINDS:
            return False
        if not GATHER_CLASSES_BY_KIND[cast("ResourceKind", spec.subject)] & state.visible_classes:
            return False
    return spec.kind != "handoff" or state.own_army_present


def feasible_candidates(state: PolicyState) -> tuple[ActionCandidate, ...]:
    return tuple(candidate_for(spec) for spec in SPECS if eligible(spec, state))


def find_candidate(
    candidates: tuple[ActionCandidate, ...], candidate_id: str
) -> ActionCandidate | None:
    return next((candidate for candidate in candidates if candidate.id == candidate_id), None)


def candidate_for(spec: ActionSpec) -> ActionCandidate:
    commands: tuple[dict[str, object], ...]
    if spec.kind == "build":
        commands = (
            {
                "type": "build",
                "menu": spec.menu,
                "building_key": spec.key,
                "intent": f"Build {spec.subject.replace('_', ' ')} (TypeSafe)",
            },
        )
    elif spec.kind == "research":
        commands = ({"type": "research", "tech": spec.subject, "intent": spec.id},)
    elif spec.kind == "train":
        commands = (
            ({"type": "queue_villager", "intent": spec.id},)
            if spec.subject == "villager"
            else ({"type": "train_unit", "unit": spec.subject, "intent": spec.id},)
        )
    elif spec.kind == "assign":
        commands = ({"type": "assign_idle", "resource": spec.subject, "intent": spec.id},)
    else:
        commands = ()
    return ActionCandidate(
        id=spec.id,
        description=spec.description,
        cost=MappingProxyType(dict(spec.cost)),
        actions=tuple(MappingProxyType(dict(command)) for command in commands),
    )


def candidate_by_id(candidate_id: str, state: PolicyState) -> ActionCandidate | None:
    spec = BY_ID.get(candidate_id)
    return candidate_for(spec) if spec is not None and eligible(spec, state) else None


__all__ = [
    "ActionCandidate",
    "CandidateId",
    "candidate_by_id",
    "eligible",
    "feasible_candidates",
    "find_candidate",
]
