"""Deterministic preferences over the same feasible catalog shown to TypeSafe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .allocation import Allocation, for_state, next_kind
from .candidates import housing_needed

if TYPE_CHECKING:
    from .advice import PolicyGoal
    from .candidates import ActionCandidate
    from .state import PolicyState


_AGE_PATH: dict[str, tuple[str, ...]] = {
    "Dark Age": ("advance_to_feudal", "build_mill", "build_lumber_camp"),
    "Feudal Age": (
        "advance_to_castle",
        "build_blacksmith",
        "build_market",
    ),
    "Castle Age": ("advance_to_imperial", "build_siege_workshop", "build_monastery"),
    "Imperial Age": (),
}
_TRAINING: dict[str, tuple[str, ...]] = {
    "Dark Age": ("train_militia",),
    "Feudal Age": ("train_spearman", "train_archer", "train_skirmisher", "train_scout_cavalry"),
    "Castle Age": ("train_knight", "train_archer", "train_spearman"),
    "Imperial Age": ("train_knight", "train_archer", "train_spearman"),
}
_VILLAGER_PREFERENCE: dict[str, int] = {"Dark Age": 30, "Feudal Age": 35}
_MILITARY_PATH = (
    "build_barracks",
    "build_archery_range",
    "build_stable",
)


def priority_action(
    candidates: tuple[ActionCandidate, ...], state: PolicyState
) -> ActionCandidate | None:
    """Correct measurable stalls only in the deterministic fallback path."""
    by_id = {candidate.id: candidate for candidate in candidates}
    if (
        state.tc_stalled
        and state.villagers is not None
        and state.pending_villagers == 0
        and (villager := by_id.get("queue_villager"))
    ):
        return villager
    if state.food_stalled:
        for action_id in ("assign_food", "build_farm", "build_mill"):
            if candidate := by_id.get(action_id):
                return candidate
    return None


def select_fallback(
    candidates: tuple[ActionCandidate, ...],
    state: PolicyState,
    allocation: Allocation | None,
    goals: tuple[PolicyGoal, ...] = (),
) -> ActionCandidate:
    """Choose one useful action; this function never changes feasibility."""
    by_id = {candidate.id: candidate for candidate in candidates}
    if priority := priority_action(candidates, state):
        return priority
    for goal in sorted(goals, key=lambda item: item.priority, reverse=True):
        if goal.progress >= 1.0:
            continue
        metric = goal.metric
        if metric == "villagers" and state.villagers is None:
            continue
        if metric == "food_workers" and "food" not in state.villager_jobs:
            continue
        preferred = {
            "villagers": ("queue_villager", "build_house", "assign_food"),
            "food_workers": ("assign_food", "build_farm", "build_mill", "assign_wood"),
            "age": _AGE_PATH.get(state.age, ()),
        }.get(metric, ())
        for action_id in preferred:
            if candidate := by_id.get(action_id):
                return candidate
    if state.idle_present:
        mix = for_state(state, allocation)
        desired = next_kind(mix, state.villager_jobs)
        if desired == "food" and "assign_food" not in by_id and "build_farm" in by_id:
            return by_id["build_farm"]
        for resource in (desired, "wood", "food", "gold", "stone"):
            if candidate := by_id.get(f"assign_{resource}"):
                return candidate

    if housing_needed(state) and (house := by_id.get("build_house")):
        return house
    for action_id in _AGE_PATH.get(state.age, ()):
        if candidate := by_id.get(action_id):
            return candidate
    target = _VILLAGER_PREFERENCE.get(state.age)
    if (target is None or state.villagers_ordered < target) and (
        villager := by_id.get("queue_villager")
    ):
        return villager
    for action_id in _MILITARY_PATH:
        if candidate := by_id.get(action_id):
            return candidate
    for action_id in _TRAINING.get(state.age, ()):
        if candidate := by_id.get(action_id):
            return candidate
    for action_id in (
        "research_double_bit_axe",
        "research_horse_collar",
        "research_gold_mining",
        "research_loom",
        "research_wheelbarrow",
    ):
        if candidate := by_id.get(action_id):
            return candidate
    return by_id["wait"]


__all__ = ["select_fallback"]
