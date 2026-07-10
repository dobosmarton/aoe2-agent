"""S5 — deterministic reactive tier.

Pure functions that emit routine actions every turn with NO LLM call. The slow
LLM tier owns strategy and combat; this fast tier handles the obvious economy
upkeep (keep the Town Center producing villagers, put idle villagers on
resources).

Design notes:
  - On alarm we return nothing and cede the turn to the LLM combat path —
    auto-garrison / town-bell here previously collapsed the economy.
  - Idle-villager assignment is DISTRIBUTED, not blanket. The HUD idle-villager
    badge glows yellow when villagers are idle (read into `state.idle_present`);
    when it does we pull a few one at a time with `.` (select next idle) and route
    each to a resource chosen by an age-keyed pattern, so newly-idle villagers
    spread across food/wood/gold instead of all piling onto one tile. This replaces
    the old `Shift-.` (select ALL idle → one right-click) blanket, which sent
    everyone to the same spot and moved the camera every turn even when nobody was
    idle. The badge's COUNT digit (template OCR, `state.idle_count`) sizes the
    batch exactly when readable; the presence colour stays the gate and the
    fallback (fixed small batch, badge re-read each turn tells us when to stop —
    it greys out once all idle are assigned). The executor resolves `target_class`
    from its detected-entity cache, so we only name the concrete resource class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entity_utils import RESOURCE_KINDS, ResourceKind, extract_attrs, nearest_class_of_kind

if TYPE_CHECKING:
    from .memory import GameState

# Stop queuing villagers past these per-age caps to bank food for aging up.
_POP_CAP_BY_AGE: dict[str, int] = {"Dark Age": 22, "Feudal Age": 35}

# Age-keyed repeating target pattern for routing idle villagers. Cycling the
# pattern yields the per-age gather ratio (e.g. Dark Age 3:2 food:wood). Seeding the
# phase on population (below) rotates the choice as villagers are produced, so even
# a lone idle villager per turn still spreads across kinds over the game.
_IDLE_PATTERN_BY_AGE: dict[str, tuple[ResourceKind, ...]] = {
    "Dark Age": ("food", "food", "food", "wood", "wood"),
    "Feudal Age": ("food", "food", "wood", "wood", "gold"),
    "Castle Age": ("food", "wood", "gold", "food", "gold", "stone"),
    "Imperial Age": ("food", "wood", "gold", "gold", "stone"),
}
_DEFAULT_IDLE_PATTERN: tuple[ResourceKind, ...] = _IDLE_PATTERN_BY_AGE["Dark Age"]

# Idle villagers dispatched per turn while the badge shows present. Each `.` costs a
# camera move + rescan, so keep this small; the badge re-read next turn drains any
# remainder. Used when only presence (not the count) is known — a `.` beyond the
# last idle villager is a harmless no-op; this is the max we'll spend chasing a
# lit badge blind.
_IDLE_DISPATCH_PER_TURN = 3
# With the badge count read (state.idle_count), the batch is sized exactly — but
# still capped so a mass-idle event (post-combat, town-bell recovery) doesn't blow
# the turn's action budget; the re-read next turn drains the rest with urgency.
_IDLE_DISPATCH_MAX = 6


def decide(entities: list[object], state: GameState, alarm: bool) -> list[dict[str, object]]:
    """Return routine action dicts for this turn (empty on alarm)."""
    if alarm:
        return []
    actions: list[dict[str, object]] = []
    actions.extend(_queue_villager_actions(state))
    actions.extend(_distribute_idle_actions(entities, state))
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


def _distribute_idle_actions(entities: list[object], state: GameState) -> list[dict[str, object]]:
    """Route idle villagers one at a time, spread across resources by age pattern.

    Gated on the HUD badge presence (`state.idle_present`): False = none idle,
    None = badge unread — both skip (never dispatch on an unknown reading). The
    batch is sized by the badge count when the digit was readable (capped at
    `_IDLE_DISPATCH_MAX`), else a blind `_IDLE_DISPATCH_PER_TURN`; each villager
    is pulled with `.` (select next idle) and right-clicked onto a resource whose
    kind is chosen by the age pattern. The badge re-read next turn drains any
    remainder (a `.` past the last idle villager is a harmless no-op).
    """
    if not state.idle_present:
        return []
    if state.idle_count is not None:
        batch = min(state.idle_count, _IDLE_DISPATCH_MAX)
    else:
        batch = _IDLE_DISPATCH_PER_TURN
    if batch == 0:  # digit says none idle — presence colour was a false positive
        return []

    pattern = _IDLE_PATTERN_BY_AGE.get(state.current_age, _DEFAULT_IDLE_PATTERN)
    origin = _tc_origin(entities)
    actions: list[dict[str, object]] = []
    for i in range(batch):
        kind = pattern[(state.population + i) % len(pattern)]
        target = _resolve_idle_target(entities, kind, origin)
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


def _tc_origin(entities: list[object]) -> tuple[float, float]:
    """Town Center center if detected, else the origin — the distance anchor."""
    for a in (extract_attrs(e) for e in entities):
        if a.class_name == "town_center":
            return a.center
    return (0.0, 0.0)


def _resolve_idle_target(
    entities: list[object], kind: ResourceKind, origin: tuple[float, float]
) -> str | None:
    """Concrete class for `kind`, falling through gather priority to any visible.

    Prefer the requested kind; if none of it is on screen, walk the gather-priority
    order (food > wood > gold > stone) so an idle villager is never wasted when
    *some* resource is visible. None only when nothing gatherable is detected.
    """
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
