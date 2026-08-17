"""Target villager allocation, and the shortfall routing that enforces it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..entity_utils import RESOURCE_KINDS, ResourceKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .state import PolicyState

# Seeded per-age targets, converted from the reactive tier's gather patterns
# (knowledge/seed/rules/allocation.yaml). Used until the strategist answers —
# on turn 1, and for the whole game when the LLM is down.
_SEED_BY_AGE: dict[str, dict[ResourceKind, int]] = {
    "Dark Age": {"food": 3, "wood": 2, "gold": 0, "stone": 0},
    "Feudal Age": {"food": 2, "wood": 2, "gold": 1, "stone": 0},
    "Castle Age": {"food": 2, "wood": 1, "gold": 2, "stone": 1},
    "Imperial Age": {"food": 1, "wood": 1, "gold": 2, "stone": 1},
}

_FOOD_CRISIS_THRESHOLD = 60
# A farm's cost plus margin. Below this the famine override keeps a wood slot,
# or the farm economy that ENDS the famine becomes unreachable (F-21).
_FARM_AFFORDABLE_WOOD = 80
_CASTLE_GOLD_COST = 200


@dataclass(frozen=True, slots=True)
class Allocation:
    """How many villagers each resource should have."""

    targets: Mapping[ResourceKind, int]

    def share(self, kind: ResourceKind) -> float:
        """This resource's fraction of the total target; 0.0 when unallocated."""
        total = sum(self.targets.values())
        return self.targets.get(kind, 0) / total if total else 0.0


def seeded(age: str) -> Allocation:
    """The default target for `age`, before the strategist has said anything."""
    return Allocation(targets=_SEED_BY_AGE.get(age, _SEED_BY_AGE["Dark Age"]))


def is_famine(state: PolicyState) -> bool:
    """Whether food is short enough to override everything else.

    A famine also suspends the wood bank: banking toward a building while
    villagers starve is how run 1 lost its production (F-8). The famine mix
    below already reserves the wood a farm needs.
    """
    return state.food < _FOOD_CRISIS_THRESHOLD


def for_state(
    state: PolicyState, strategist: Allocation | None, wood_target: int | None = None
) -> Allocation:
    """The target in force this tick, with the famine and bank overrides applied.

    A famine outranks everything, and its two branches are a matched pair whose
    order is load-bearing: all-food starved wood to 0 and locked out the farming
    that ends the famine (F-8 then F-21).

    A pending build the stock cannot cover adds ONE wood slot — not every slot.
    Banking to the exclusion of food is the same failure in another costume.
    """
    if is_famine(state):
        if state.wood < _FARM_AFFORDABLE_WOOD:
            return Allocation(targets={"food": 2, "wood": 1, "gold": 0, "stone": 0})
        return Allocation(targets={"food": 1, "wood": 0, "gold": 0, "stone": 0})

    base = strategist or seeded(state.age)
    if wood_target is not None and state.wood < wood_target:
        base = _with_extra(base, "wood")
    if state.age == "Feudal Age" and state.gold < _CASTLE_GOLD_COST:
        base = _with_extra(base, "gold")
    return base


def _with_extra(base: Allocation, kind: ResourceKind) -> Allocation:
    """`base` with one more villager wanted on `kind`."""
    targets = dict(base.targets)
    targets[kind] = targets.get(kind, 0) + 1
    return Allocation(targets=targets)


def next_kind(allocation: Allocation, jobs: Mapping[str, int]) -> ResourceKind:
    """The resource furthest below its target share of the current workforce.

    Ties break by the fixed `RESOURCE_KINDS` order so routing is deterministic.
    Callers routing a batch must fold each choice back into `jobs`, or the whole
    batch goes to the same resource.
    """
    staffed = sum(jobs.get(kind, 0) for kind in RESOURCE_KINDS)
    return min(
        RESOURCE_KINDS,
        key=lambda kind: (jobs.get(kind, 0) - allocation.share(kind) * staffed, kind),
    )


def with_one_more(jobs: Mapping[str, int], kind: ResourceKind) -> dict[str, int]:
    """`jobs` with `kind` incremented — the fold for routing a batch."""
    return {**jobs, kind: jobs.get(kind, 0) + 1}
