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

from .entity_utils import (
    RESOURCE_KINDS,
    ResourceKind,
    first_center_of_class,
    nearest_class_of_kind,
)

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
# Count trust gate: after this many consecutive turns with the badge lit, a count
# smaller than the blind batch is treated as an OCR under-read (a working dispatch
# drains the badge within a turn or two; a lit badge that outlives its own count is
# the digit lying). Presence is the robust signal, so never dispatch *less* than
# the blind batch on a long streak — extra `.` presses past the last idle villager
# are harmless no-ops.
_IDLE_COUNT_SUSPECT_STREAK = 4
# Econ build-menu key for a Farm (executor.BUILD_KEY_TO_CLASS) — emitted when a
# food turn finds nothing huntable/foragable on screen. Farms are never gather
# targets (see entity_utils.GATHER_CLASSES_BY_KIND): each supports exactly one
# villager and misdetected bare-ground "farms" strand villagers, so the rule is
# one FRESH farm per idle villager — the builder auto-farms the field it
# finishes. The executor's build gates (mill prerequisite, wood cost, via
# build_rejection) reject the action at zero keystroke cost when it can't work.
_FARM_BUILD_KEY = "a"

# Feudal Age economics. Research costs 500 food and happens at the TC (`h` →
# `z`, prompts/hotkeys.md). With the villager queue firing every turn, 50 food
# at a time, 500 can never accumulate (2026-07-11 run 3, F-16) — so once the
# Dark Age economy is established (`_FEUDAL_BANK_POP` villagers) the queue
# stops and food BANKS toward the research. The two Dark Age buildings Feudal
# also requires are always up by then (houses + mill).
_FEUDAL_FOOD_COST = 500
_FEUDAL_BANK_POP = 16
# Below this food, every idle slot is forced to food: honoring the rotation's
# wood/gold slots during a famine starves villager production (run 1, F-8).
_FOOD_CRISIS_THRESHOLD = 60


def decide(entities: list[object], state: GameState, alarm: bool) -> list[dict[str, object]]:
    """Return routine action dicts for this turn (empty on alarm)."""
    if alarm:
        return []
    actions: list[dict[str, object]] = []
    actions.extend(_age_up_actions(state))
    actions.extend(_queue_villager_actions(state))
    actions.extend(_distribute_idle_actions(entities, state))
    return actions


def _food(state: GameState) -> int:
    """Last-known food reading, defensively coerced.

    The runtime guard stays even though resources is typed dict[str, int]:
    values arrive across the untyped LLM-observation boundary, so the
    annotation is a claim the data can't be trusted to honor.
    """
    try:
        return int(state.resources.get("food", 0))
    except (TypeError, ValueError):
        return 0


def _age_up_actions(state: GameState) -> list[dict[str, object]]:
    """Research Feudal Age the moment the food is banked (Dark Age only).

    Runs before the villager queue so the 500 food buys the age, not another
    villager. Pressing while the button is unavailable (already researching,
    building requirements missing) is a harmless no-op, and once research
    starts the food drops below the threshold, so this doesn't spam.
    """
    if state.current_age != "Dark Age" or _food(state) < _FEUDAL_FOOD_COST:
        return []
    return [
        {"type": "press", "key": "h", "intent": "Select TC (age up)"},
        {"type": "press", "key": "z", "intent": "Research Feudal Age (reactive)"},
    ]


def _banking_for_feudal(state: GameState) -> bool:
    """Dark Age with an established economy: stop buying villagers, bank 500."""
    return state.current_age == "Dark Age" and state.population >= _FEUDAL_BANK_POP


def _pop_below_cap(state: GameState) -> bool:
    age_cap = _POP_CAP_BY_AGE.get(state.current_age, state.population_cap)
    return state.population < min(state.population_cap, age_cap)


def _queue_villager_actions(state: GameState) -> list[dict[str, object]]:
    if _banking_for_feudal(state) or not _pop_below_cap(state):
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
    `_IDLE_DISPATCH_MAX`, and floored at the blind batch once the badge has been
    lit `_IDLE_COUNT_SUSPECT_STREAK` turns — see the trust-gate note above), else
    a blind `_IDLE_DISPATCH_PER_TURN`; each villager
    is pulled with `.` (select next idle) and right-clicked onto a resource whose
    kind is chosen by the age pattern. A food turn with nothing huntable or
    foragable on screen builds a fresh farm instead (see `_FARM_BUILD_KEY`).
    The badge re-read next turn drains any remainder (a `.` past the last idle
    villager is a harmless no-op).
    """
    if not state.idle_present:
        return []
    batch = _idle_batch_size(state)
    if batch == 0:  # digit says none idle — presence colour was a false positive
        return []

    pattern = _idle_pattern(state)
    origin = _tc_origin(entities)
    actions: list[dict[str, object]] = []
    farm_queued = False
    for i in range(batch):
        kind = pattern[(state.population + i) % len(pattern)]
        if kind == "food" and not farm_queued and nearest_class_of_kind(entities, "food") is None:
            # Food wanted but nothing huntable/foragable on screen: build a fresh
            # farm for this villager instead of falling through to wood. One per
            # turn — the HUD snapshot the build gate checks doesn't see this
            # turn's spend, so a second build here couldn't be cost-checked.
            actions.append(
                {
                    "type": "build",
                    "building_key": _FARM_BUILD_KEY,
                    "intent": "Build farm for idle villager (no forage/huntables visible)",
                }
            )
            farm_queued = True
            continue
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


def _idle_batch_size(state: GameState) -> int:
    """Dispatches this turn: the badge count when trusted, else the blind batch.

    0 means the digit read "none idle" (the presence colour was a false
    positive). The count is floored at the blind batch once the badge has been
    lit `_IDLE_COUNT_SUSPECT_STREAK` consecutive turns — a lit badge outliving
    its own small count means the digit is under-reading.
    """
    if state.idle_count is None:
        return _IDLE_DISPATCH_PER_TURN
    batch = min(state.idle_count, _IDLE_DISPATCH_MAX)
    if state.idle_streak >= _IDLE_COUNT_SUSPECT_STREAK:
        batch = max(batch, _IDLE_DISPATCH_PER_TURN)
    return batch


def _idle_pattern(state: GameState) -> tuple[ResourceKind, ...]:
    """Age-keyed gather rotation — overridden to all-food during a famine
    (wood/gold can wait; villager production and the Feudal bank cannot)."""
    if _food(state) < _FOOD_CRISIS_THRESHOLD:
        return ("food",)
    return _IDLE_PATTERN_BY_AGE.get(state.current_age, _DEFAULT_IDLE_PATTERN)


def _tc_origin(entities: list[object]) -> tuple[float, float]:
    """Town Center center if detected, else the origin — the distance anchor."""
    return first_center_of_class(entities, "town_center") or (0.0, 0.0)


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
