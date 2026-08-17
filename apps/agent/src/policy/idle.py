"""Idle-villager dispatch — procedural because it reads the entity list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..entity_utils import (
    RESOURCE_KINDS,
    ResourceKind,
    first_center_of_class,
    nearest_class_of_kind,
)
from . import allocation

if TYPE_CHECKING:
    from .state import PolicyState

# Dispatches per turn when only badge PRESENCE is known. Each `.` costs a camera
# move; the badge re-read next turn drains any remainder.
_IDLE_DISPATCH_PER_TURN = 3
# Cap when the badge COUNT is readable, so a mass-idle event cannot blow the
# turn's action budget.
_IDLE_DISPATCH_MAX = 6
# A lit badge outliving its own count means the digit is under-reading (F-4).
_IDLE_COUNT_SUSPECT_STREAK = 4

# A food slot with nothing huntable on screen builds a fresh farm instead.
# Farms are never gather targets — see entity_utils.GATHER_CLASSES_BY_KIND.
_FARM_BUILD_KEY = "a"

# The farm is the only build this module costs itself; every other build cost
# comes from the rule that emits it (see engine.wood_bank_target).
_FARM_WOOD_COST = 60
# Headroom above a bank target so a purchase does not land on the cost boundary.
_WOOD_BANK_MARGIN = 20


def distribute_idle(
    entities: list[object],
    state: PolicyState,
    wood_target: int | None,
    strategist_allocation: allocation.Allocation | None = None,
) -> list[dict[str, object]]:
    """Route idle villagers one at a time toward the target allocation.

    Gated on badge presence: False means none idle, None means unread — both
    skip, so an unknown reading never triggers a camera move.
    """
    if not state.idle_present:
        return []
    batch = _idle_batch_size(state)
    if batch == 0:  # the digit says none idle — the colour was a false positive
        return []

    target_mix = allocation.for_state(state, strategist_allocation, wood_target)
    jobs = dict(state.villager_jobs)
    origin = _tc_origin(entities)
    actions: list[dict[str, object]] = []
    farm_queued = False
    for _ in range(batch):
        kind = allocation.next_kind(target_mix, jobs)
        jobs = allocation.with_one_more(jobs, kind)
        if kind == "food" and not farm_queued and nearest_class_of_kind(entities, "food") is None:
            # One per turn: the HUD snapshot the build gate checks cannot see
            # this turn's spend, so a second build could not be cost-checked.
            actions.append(
                {
                    "type": "build",
                    "building_key": _FARM_BUILD_KEY,
                    "intent": "Build farm for idle villager (no forage/huntables visible)",
                }
            )
            farm_queued = True
            continue
        target = resolve_idle_target(entities, kind, origin)
        if target is None:
            break  # nothing gatherable on screen — retry next turn
        actions.append(
            {
                "type": "press",
                "key": ".",
                "rescan": True,
                "intent": f"Select next idle villager → {kind}",
            }
        )
        actions.append(
            {
                "type": "right_click",
                "target_class": target,
                "intent": f"Send idle villager to {target} ({kind})",
            }
        )
    return actions


def _idle_batch_size(state: PolicyState) -> int:
    """The badge count when trusted, else the blind batch. 0 means none idle."""
    if state.idle_count is None:
        return _IDLE_DISPATCH_PER_TURN
    batch = min(state.idle_count, _IDLE_DISPATCH_MAX)
    if state.idle_streak >= _IDLE_COUNT_SUSPECT_STREAK:
        batch = max(batch, _IDLE_DISPATCH_PER_TURN)
    return batch


def farm_bank_target() -> int:
    """The fallback target once a mill stands.

    The farm is emitted by this module, not by a rule, so it has no registry
    entry to derive a cost from.
    """
    return _FARM_WOOD_COST + _WOOD_BANK_MARGIN


def _tc_origin(entities: list[object]) -> tuple[float, float]:
    """Town Center centre if detected, else the origin — the distance anchor."""
    return first_center_of_class(entities, "town_center") or (0.0, 0.0)


def resolve_idle_target(
    entities: list[object], kind: ResourceKind, origin: tuple[float, float]
) -> str | None:
    """The requested kind if visible, else the next by gather priority."""
    target = nearest_class_of_kind(entities, kind, origin)
    if target is not None:
        return target
    for fallback in RESOURCE_KINDS:
        if fallback == kind:
            continue
        target = nearest_class_of_kind(entities, fallback, origin)
        if target is not None:
            return target
    return None
