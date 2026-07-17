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

# Stop ORDERING villagers past these per-age targets so food banks for aging
# up. Gated on GameState.villagers_ordered, never on the delivered HUD
# population: orders lead it by the TC queue depth (~25 s per villager vs a
# ~10 s turn), and a population brake over-delivered 40 villagers whose cost
# WAS the Feudal bank (run 11, F-38). Dark Age 30 is the user directive; a
# drift test pins it to the executor's order gate, which backstops
# LLM-initiated queues with the same number.
_VILLAGER_TARGET_BY_AGE: dict[str, int] = {"Dark Age": 30, "Feudal Age": 35}

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
# age's villager target is ordered the queue stops and food BANKS toward the
# research.
_FEUDAL_FOOD_COST = 500
# Feudal also requires TWO qualifying Dark Age buildings — and houses don't
# count (run 6, F-26: 14 age-up presses no-oped against a greyed button with
# only the mill built while 767 food sat banked). Mirrors
# evaluation.world_sim.FEUDAL_PREREQ_BUILDINGS; a drift test pins the two.
_FEUDAL_PREREQ_CLASSES = frozenset({"mill", "lumber_camp"})
# Start building the prerequisites a bit before the banking phase (pop 16) so
# they're standing by the time 500 food is.
_FEUDAL_PREP_POP = 12
_MILL_BUILD_KEY = "w"  # econ menu: Mill (100 wood)
_LUMBER_CAMP_BUILD_KEY = "r"  # econ menu: Lumber Camp (100 wood)
# Below this food, the idle rotation is overridden toward food: honoring the
# normal wood/gold slots during a famine starves villager production (run 1, F-8).
_FOOD_CRISIS_THRESHOLD = 60
# A farm's wood cost (mirrors executor._BUILD_WOOD_COST["a"]; duplicated so the
# reactive tier stays dependency-free — a drift test is the cross-check, V-4).
# During a famine the override must NOT starve wood below this: run 4 (F-21)
# showed all-food routing pinning wood at 0, locking out the farm economy the
# famine needed to end.
_FARM_WOOD_COST = 60
# A lumber camp's wood cost (mirrors executor._BUILD_WOOD_COST["r"], same
# drift-test arrangement as the farm cost above). Run 8 (F-34): the wood bank
# targeted only the farm, so the camp — the second Feudal prerequisite — was
# rejected 19 times at 37-79 wood and Feudal stayed unreachable.
_LUMBER_CAMP_WOOD_COST = 100
# A mill's wood cost (mirrors executor._BUILD_WOOD_COST["w"], same drift test).
# The mill is the OTHER Feudal prerequisite AND the farm unlock, so the wood
# bank must reach it too — run 12 (F-41) starved because the reactive tier had
# no mill rule at all and the executor (its only mill-builder) was down.
_MILL_WOOD_COST = 100
# Headroom above a bank target so a purchase doesn't leave the stock exactly
# at the cost boundary. Run 5 (F-23): six farm attempts failed at wood 48-59.
# A separate constant so the V-4 drift tests keep pinning the raw costs to the
# executor's table.
_WOOD_BANK_MARGIN = 20


def decide(entities: list[object], state: GameState, alarm: bool) -> list[dict[str, object]]:
    """Return routine action dicts for this turn (empty on alarm)."""
    if alarm:
        return []
    actions: list[dict[str, object]] = []
    actions.extend(_age_up_actions(state))
    actions.extend(_queue_villager_actions(state))
    actions.extend(_feudal_prep_actions(state))
    actions.extend(_distribute_idle_actions(entities, state))
    return actions


def _resource(state: GameState, kind: ResourceKind) -> int:
    """Last-known reading for `kind`, defensively coerced.

    The runtime guard stays even though resources is typed dict[str, int]:
    values arrive across the untyped LLM-observation boundary, so the
    annotation is a claim the data can't be trusted to honor.
    """
    try:
        return int(state.resources.get(kind, 0))
    except (TypeError, ValueError):
        return 0


def _age_up_actions(state: GameState) -> list[dict[str, object]]:
    """Research Feudal Age once the food is banked AND the buildings qualify.

    Runs before the villager queue so the 500 food buys the age, not another
    villager. Gated on the two-building requirement being visibly met, so the
    press fires once when it can succeed instead of spamming no-ops (run 6:
    14 presses against a greyed button — and each press was a chance for the
    F-27 UI-context leak). Selecting the TC (`h`) itself clears any open build
    menu / placement ghost by switching selection, so `z` can't land in the
    econ menu (where Z = Outpost); an escape prefix here OPENED the game menu
    when nothing needed canceling (run 8, F-32).
    """
    if (
        state.current_age != "Dark Age"
        or _resource(state, "food") < _FEUDAL_FOOD_COST
        or not _FEUDAL_PREREQ_CLASSES.issubset(state.buildings_seen)
    ):
        return []
    return [
        {"type": "press", "key": "h", "intent": "Select TC (age up)"},
        {"type": "press", "key": "z", "intent": "Research Feudal Age (reactive)"},
    ]


def _needs_mill(state: GameState) -> bool:
    """Feudal prep pending: Dark Age economy established, no mill standing.

    The mill is BOTH a Feudal prerequisite and the farm unlock — the entire
    late-Dark-Age food engine hangs off it. Run 12 (F-41) had the executor
    down for 85/95 turns and starved, because a mill had only ever been built
    by the LLM; the reactive tier knew the lumber camp but not the mill. This
    gives the fast tier its own path to the food engine.
    """
    return (
        state.current_age == "Dark Age"
        and state.population >= _FEUDAL_PREP_POP
        and "mill" not in state.buildings_seen
    )


def _needs_lumber_camp(state: GameState) -> bool:
    """Feudal prep pending: Dark Age economy established, no camp standing."""
    return (
        state.current_age == "Dark Age"
        and state.population >= _FEUDAL_PREP_POP
        and "lumber_camp" not in state.buildings_seen
    )


def _feudal_prep_actions(state: GameState) -> list[dict[str, object]]:
    """Ensure both qualifying Dark Age buildings exist — MILL FIRST, then camp.

    The mill leads when neither stands: it unlocks farms (the food engine) on
    top of counting toward Feudal, so a reactive-only game (executor down, run
    12 F-41) can still feed itself. One build per turn — the executor's
    unique-building gate rejects re-emits once the class is confirmed or
    pending, its cost gate waits out the 100 wood, and the circuit breaker
    suppresses a placement that keeps vanishing. The lumber camp doubles as a
    wood-income boost (closer drop-off).
    """
    if _needs_mill(state):
        return [
            {
                "type": "build",
                "building_key": _MILL_BUILD_KEY,
                "intent": "Build mill (Feudal prerequisite + farm/food unlock)",
            }
        ]
    if _needs_lumber_camp(state):
        return [
            {
                "type": "build",
                "building_key": _LUMBER_CAMP_BUILD_KEY,
                "intent": "Build lumber camp (Feudal prerequisite + wood income)",
            }
        ]
    return []


def _orders_below_target(state: GameState) -> bool:
    """Whether another villager order fits the age target and the pop cap."""
    target = _VILLAGER_TARGET_BY_AGE.get(state.current_age, state.population_cap)
    return state.villagers_ordered < min(state.population_cap, target)


def _queue_villager_actions(state: GameState) -> list[dict[str, object]]:
    """One villager order per turn while below the age target.

    Emitted as the first-class `queue_villager` action so the executor's
    order ledger counts it and its gate re-checks target + food (F-38).
    """
    if not _orders_below_target(state):
        return []
    return [{"type": "queue_villager", "intent": "Queue villager (reactive)"}]


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


def _wood_bank_target(state: GameState) -> int | None:
    """Wood the rotation should bank toward: the binding build goal's cost.

    The Feudal prerequisites (mill first, then lumber camp) outrank farms; each
    target carries the margin so a purchase doesn't leave the stock exactly at
    the boundary. Run 8 (F-34): a farm-only target let wood plateau at 65 while
    the camp cost 100. Ordered to match `_feudal_prep_actions` (mill leads, it
    unlocks farms). None when no wood-gated goal is pending.
    """
    if _needs_mill(state):
        return _MILL_WOOD_COST + _WOOD_BANK_MARGIN
    if _needs_lumber_camp(state):
        return _LUMBER_CAMP_WOOD_COST + _WOOD_BANK_MARGIN
    if "mill" in state.buildings_seen:
        return _FARM_WOOD_COST + _WOOD_BANK_MARGIN
    return None


def _idle_pattern(state: GameState) -> tuple[ResourceKind, ...]:
    """Age-keyed gather rotation, overridden during a food famine.

    Famine with farming affordable → all-food (wood/gold can wait; villager
    production and the Feudal bank cannot). Famine with wood below a farm's
    cost plus margin → 2:1 food:wood, so the farm economy that ENDS the famine
    stays reachable — the pure all-food override starved wood to 0 and locked
    the loop shut (run 4, F-21). Outside a famine, wood below the binding
    goal's bank target gets one extra wood slot in the rotation (F-23, F-34).
    """
    if _resource(state, "food") < _FOOD_CRISIS_THRESHOLD:
        if _resource(state, "wood") < _FARM_WOOD_COST + _WOOD_BANK_MARGIN:
            return ("food", "food", "wood")
        return ("food",)
    pattern = _IDLE_PATTERN_BY_AGE.get(state.current_age, _DEFAULT_IDLE_PATTERN)
    target = _wood_bank_target(state)
    if target is not None and _resource(state, "wood") < target:
        return ("wood", *pattern)
    return pattern


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
