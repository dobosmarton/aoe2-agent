"""S5 — deterministic reactive tier.

Pure functions that emit routine actions every turn with NO LLM call. The slow
LLM tier owns strategy and combat; this fast tier handles the obvious economy
upkeep (keep the Town Center producing villagers, put idle villagers on
resources).

Design notes:
  - On alarm we return nothing and cede the turn to the LLM combat path —
    auto-garrison / town-bell here previously collapsed the economy.
  - Idle-villager assignment uses the game's own `Shift-.` (select ALL idle
    villagers) hotkey followed by one right-click, dispatching the whole idle
    pool in two actions. It selects nothing when none are idle, so the
    right-click is a safe no-op. Selecting all sidesteps the "how many are
    idle" question that YOLO cannot answer, and fits the per-turn commit cap
    (a single `.` cycled only one villager, leaving the rest waiting). The
    executor resolves `target_class` from its detected-entity cache, so we only
    name the nearest resource class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entity_utils import extract_attrs

if TYPE_CHECKING:
    from .memory import GameState

# Resource classes a villager can gather from, in assignment priority order.
_RESOURCE_TIERS: tuple[frozenset[str], ...] = (
    frozenset({"sheep", "boar", "deer", "berry_bush", "farm"}),  # food first
    frozenset({"tree"}),  # then wood
    frozenset({"gold_mine"}),
    frozenset({"stone_mine"}),
)

# Stop queuing villagers past these per-age caps to bank food for aging up.
_POP_CAP_BY_AGE: dict[str, int] = {"Dark Age": 22, "Feudal Age": 35}


def decide(entities: list[object], state: GameState, alarm: bool) -> list[dict[str, object]]:
    """Return routine action dicts for this turn (empty on alarm)."""
    if alarm:
        return []
    actions: list[dict[str, object]] = []
    actions.extend(_queue_villager_actions(state))
    actions.extend(_idle_villager_actions(entities))
    return actions


def _pop_below_cap(state: GameState) -> bool:
    age_cap = _POP_CAP_BY_AGE.get(state.current_age, state.population_cap)
    return state.population < min(state.population_cap, age_cap)


def _queue_villager_actions(state: GameState) -> list[dict[str, object]]:
    if not _pop_below_cap(state):
        return []
    return [
        {"type": "press", "key": "h", "intent": "Select TC (reactive)"},
        {"type": "press", "key": "q", "intent": "Queue villager (reactive)"},
    ]


def _idle_villager_actions(entities: list[object]) -> list[dict[str, object]]:
    target = _nearest_resource_class(entities)
    if target is None:
        return []
    return [
        {
            "type": "press",
            "key": ".",
            "modifiers": ["shift"],
            "rescan": True,
            "intent": "Select ALL idle villagers (reactive)",
        },
        {
            "type": "right_click",
            "target_class": target,
            "intent": f"Send all idle villagers to {target} (reactive)",
        },
    ]


def _dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _nearest_resource_class(entities: list[object]) -> str | None:
    """Class of the nearest gatherable resource (food > wood > gold > stone).

    "Nearest" is measured to the Town Center if one is detected, else the origin.
    Returns None when no resource is visible.
    """
    attrs = [extract_attrs(e) for e in entities]
    origin = next((a.center for a in attrs if a.class_name == "town_center"), (0.0, 0.0))
    for tier in _RESOURCE_TIERS:
        candidates = [a for a in attrs if a.class_name in tier]
        if candidates:
            return min(candidates, key=lambda a: _dist_sq(a.center, origin)).class_name
    return None
