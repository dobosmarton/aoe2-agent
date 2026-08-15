"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pyautogui
import structlog
from pydantic import BaseModel

from .config import config
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.stdlib.get_logger()

# Configure pyautogui for better game compatibility
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.02

# Module-level state (updated per-action batch)
_window_offset: tuple[int, int] = (0, 0)
_detected_entities: list[dict] = []
_rescan_fn: Callable[[], Awaitable[None]] | None = None
_rescan_full_fn: Callable[[], Awaitable[None]] | None = None


@dataclass
class ActionResult:
    """Result of executing a single action."""

    success: bool
    detail: str


# ---------------------------------------------------------------------------
# Entity cache management
# ---------------------------------------------------------------------------


def set_rescan_fn(fn: Callable[[], Awaitable[None]]) -> None:
    """Set the rescan callback for mid-turn screenshot+detection."""
    global _rescan_fn
    _rescan_fn = fn


def set_rescan_full_fn(fn: Callable[[], Awaitable[None]]) -> None:
    """Set the full detection callback for thorough SAHI scan."""
    global _rescan_full_fn
    _rescan_full_fn = fn


def get_rescan_fn() -> Callable[[], Awaitable[None]] | None:
    """Return the registered fast-rescan callback, or None if unset."""
    return _rescan_fn


def set_detected_entities(entities: Sequence[object]) -> None:
    """Cache detected entities for target_id/target_class resolution.

    Accepts either DetectedEntity instances (with `.to_dict()`) or already-
    serialized dict shapes — normalizes both into the dict cache.
    """
    global _detected_entities
    normalized: list[dict] = []
    for e in entities:
        to_dict = getattr(e, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, dict):
                normalized.append(converted)
        elif isinstance(e, dict):
            normalized.append(e)
        else:
            log.warning("detected_entity_unrecognized_type", entity_type=type(e).__name__)
    _detected_entities = normalized
    # Every detection frame (turn scans and mid-turn rescans alike) feeds the
    # build-prerequisite evidence — thresholded, so a one-frame phantom can't
    # poison the gates (run 7: a misdetected mill unlocked impossible farms
    # AND blocked the real mill).
    record_building_sightings(str(e.get("class", "")) for e in normalized)
    log.debug("detected_entities_set", count=len(_detected_entities))


def get_detected_entities() -> list[dict]:
    """Return the current detected entity list."""
    return _detected_entities


def clear_detected_entities() -> None:
    """Clear the cached detected entities."""
    global _detected_entities
    _detected_entities = []


# ---------------------------------------------------------------------------
# Build gates: per-game state feeding build_rejection + placement settlement
# ---------------------------------------------------------------------------

# Only build a house when within this many pop of the cap. The 2026-07-11 run
# built houses at 15+ headroom (125 wood) while the first farm starved for 60 —
# a satisfied "raise pop cap" goal kept re-triggering because every house
# "succeeds".
_HOUSE_HEADROOM_MAX = 4
# The game's population-cap maximum: houses past it add nothing.
_GAME_POP_CAP_LIMIT = 200
# Residual noise a settlement tolerates AFTER estimated gather income is
# deducted (OCR jitter, income-estimate error). The income itself is modeled,
# not slack-covered: run 13 (F-45/T-537) had a 30-villager economy gathering
# +140 wood across a 25-wood house settlement — no fixed slack survives both
# that and a 4-villager opening.
_PLACEMENT_INCOME_SLACK = 20
# EMA weight for the per-snapshot wood-income estimate (updated only on
# windows with no pending spend). 0.5 tracks the fast income ramp of a
# growing villager count while smoothing single-frame OCR blips.
_INCOME_EMA_WEIGHT = 0.5
# Stale-OCR grace: identical wood readings a pending placement survives before
# it is settled anyway.
_PLACEMENT_SETTLE_ATTEMPTS = 3
# Distinct detection frames before a building class is REPORTED as sighted
# (context line only — sightings never gate builds: run 9, F-36, a persistent
# phantom mill beat any count threshold and 14 outposts got built through the
# unlocked farm slot).
_SIGHTING_MIN_FRAMES = 3
# Circuit breaker (T-530): consecutive missing settlements for one building
# class before its builds are suppressed, and how many HUD snapshots the
# suppression lasts. Run 9: 32 identical farm attempts each burned resources.
_MISSING_STREAK_LIMIT = 3
_MISSING_SUPPRESS_SNAPSHOTS = 5
# Villager-order ledger (T-531). Orders lead the HUD population by the TC
# queue depth (~25 s per villager vs a ~10 s turn), so a brake on DELIVERED
# population overshoots: run 11 (F-38) pressed q 36 times, every one at pop
# ≤ 15, and the queue delivered 40. The game starts with 4 villagers
# (mirrors memory.INITIAL_POPULATION; drift test pins the two).
_STARTING_VILLAGERS = 4
_VILLAGER_FOOD_COST = 50
# Villager order targets by age (T-538 — run 13 reached Feudal and the flat
# Dark Age 30 overruled the reactive tier's Feudal 35 while the rejection
# message kept teaching "bank for the Feudal Age" IN Feudal). Dark Age 30 is
# the user directive (run 11): enough economy to bank the 500-food Feudal
# cost — every order past a target IS that age's bank being spent. Ages past
# the map (Castle+) have no order cap; only the food gate applies. The
# reactive tier's _VILLAGER_TARGET_BY_AGE mirrors this map (drift test).
_VILLAGER_ORDER_TARGET_BY_AGE: dict[str, int] = {"Dark Age": 30, "Feudal Age": 35}
# What the banked resources are FOR, per age — keeps the rejection message's
# teaching age-correct. Same keys as the target map (the drift test pins it).
_NEXT_AGE: dict[str, str] = {"Dark Age": "Feudal Age", "Feudal Age": "Castle Age"}


# A placement whose foundation wasn't visually confirmed, awaiting settlement
# against the HUD wood spend (2026-07-11 run 2, F-11: YOLO can't see foundations,
# so a fresh rescan reports almost every REAL placement as failed — that false
# negative caused a duplicate mill). Mutable on purpose: settles_left counts down.
@dataclass
class _PendingPlacement:
    building_class: str
    wood_cost: int
    wood_before: int  # wood per the HUD snapshot when the placement was made
    noted_at_snapshot: int  # snapshot_count the wood_before reading belongs to
    settles_left: int = _PLACEMENT_SETTLE_ATTEMPTS


@dataclass
class _BuildGates:
    """Per-game state behind build_rejection + placement settlement.

    population/resources: this iteration's HUD reading (None = no reading yet —
    the gates then allow the build rather than blocking on missing data).
    buildings_confirmed: PURCHASE-GRADE evidence only — building classes the
    agent provably bought (wood-delta settlement) or visually verified placing.
    Detection sightings never enter here: run 7's flickering phantom poisoned
    the gates at 1 frame, and run 9's PERSISTENT phantom beat the 3-frame
    threshold too (F-36) — a mill-less econ menu then builds OUTPOSTS through
    the unlocked farm slot, so gate evidence must be self-generated.
    pending_placements: builds awaiting wood-delta settlement.
    """

    population: tuple[int, int] | None = None
    resources: dict[str, int] | None = None
    buildings_confirmed: set[str] = field(default_factory=set)
    # Frames each gate-relevant building class has been detected in —
    # informational only (the context line reports classes past
    # _SIGHTING_MIN_FRAMES as unverified sightings).
    building_sightings: dict[str, int] = field(default_factory=dict)
    pending_placements: list[_PendingPlacement] = field(default_factory=list)
    # Estimated wood gathered per snapshot window (EMA over windows with no
    # pending spend; None until the first clean window). Settlement deducts it
    # from the observed delta so gather income can't mask a purchase (T-537).
    wood_income_per_snapshot: float | None = None
    # Circuit breaker (T-530): consecutive missing settlements per class, and
    # the snapshot count until which a repeatedly-missing class stays blocked.
    missing_streaks: dict[str, int] = field(default_factory=dict)
    suppressed_until: dict[str, int] = field(default_factory=dict)
    snapshot_count: int = 0
    # Villagers ordered so far (T-531) — self-generated ground truth that
    # leads the delivered HUD population by the TC queue depth.
    villagers_ordered: int = _STARTING_VILLAGERS
    # Validated age from GameState (strategist OCR), synced once per turn —
    # selects the villager order target and the rejection message (T-538).
    current_age: str = "Dark Age"


_build_gates = _BuildGates()


def observe_hud(population: int, population_cap: int, resources: Mapping[str, int]) -> None:
    """Feed this turn's HUD reading into the build gates.

    Not just a cache write: the fresh wood value first updates the gather-income
    estimate, then SETTLES pending placements (confirming purchases / flagging
    missing ones) before the snapshot is replaced — the ordering the wood-delta
    check depends on.
    """
    _build_gates.snapshot_count += 1
    _observe_wood_income(resources.get("wood"))
    _settle_pending_placements(resources.get("wood"))
    _build_gates.population = (population, population_cap)
    _build_gates.resources = dict(resources)


def observe_age(age: str) -> None:
    """Sync the validated age (GameState, strategist OCR) into the gates.

    Selects the villager order target and the rejection message's teaching
    (T-538). Falsy input (age not yet read this game) keeps the last value —
    the gates start at Dark Age, which is always correct at game start.
    """
    if age:
        _build_gates.current_age = age


def _observe_wood_income(wood_now: int | None) -> None:
    """Update the per-snapshot wood-income EMA from a clean window.

    Only windows with NO pending placements count — a window containing a
    spend would drag the estimate down and re-open the false-missing hole the
    estimate exists to close. Negative deltas (OCR blips) clamp to 0, which
    pulls the estimate toward the safe direction (under-crediting income).
    """
    wood_before = (_build_gates.resources or {}).get("wood")
    if wood_now is None or wood_before is None or _build_gates.pending_placements:
        return
    delta = max(wood_now - wood_before, 0)
    previous = _build_gates.wood_income_per_snapshot
    if previous is None:
        _build_gates.wood_income_per_snapshot = float(delta)
    else:
        _build_gates.wood_income_per_snapshot = (
            _INCOME_EMA_WEIGHT * delta + (1 - _INCOME_EMA_WEIGHT) * previous
        )


def _expected_income(noted_at_snapshot: int) -> float:
    """Wood the economy likely gathered since a placement's baseline reading.

    Scales with elapsed snapshots so stale-OCR retries (which accumulate
    several windows of income before the reading moves) are credited fully.
    0.0 while no clean window has been observed yet — settlement then behaves
    exactly as before the income model existed.
    """
    ema = _build_gates.wood_income_per_snapshot
    if ema is None:
        return 0.0
    elapsed = max(_build_gates.snapshot_count - noted_at_snapshot, 1)
    return ema * elapsed


def _note_pending_placement(building_key: str) -> None:
    """Queue an unconfirmed placement for wood-delta settlement next snapshot."""
    cls = BUILD_KEY_TO_CLASS.get(building_key)
    cost = _BUILD_WOOD_COST.get(building_key)
    wood_before = (_build_gates.resources or {}).get("wood")
    if cls is None or cost is None or wood_before is None:
        # No wood baseline to settle against — the placement stays unconfirmed
        # for good, despite the caller's "settled next turn" detail. Say so.
        log.debug("placement_pending_dropped", building_key=building_key)
        return
    _build_gates.pending_placements.append(
        _PendingPlacement(
            building_class=cls,
            wood_cost=cost,
            wood_before=wood_before,
            noted_at_snapshot=_build_gates.snapshot_count,
        )
    )


def _settle_pending_placements(wood_now: int | None) -> None:
    """Confirm or drop pending placements using the game's own ledger — the HUD.

    A placement that consumed its wood cost DID succeed regardless of what
    detection saw (the resource bar is authoritative; the vision model can't
    see foundations). Estimated gather income since the baseline reading is
    deducted from the observed delta first — run 13 (F-45) gathered +140 wood
    across a 25-wood house settlement, so the raw delta alone judged every
    real purchase MISSING and the circuit breaker locked out five building
    classes. Judged FIFO, erring toward confirmation: a false "failed" report
    is what caused the duplicate mill, while a false success merely delays
    the retry by a turn. An unchanged wood reading is treated as stale OCR
    and re-checked next snapshot (up to `settles_left` times). Confirmed
    spend is deducted per shared baseline before judging the next entry, so
    one wood drop confirms at most one pending of a given cost — run 3
    (F-17) settled two mills off a single purchase.
    """
    if not _build_gates.pending_placements or wood_now is None:
        return
    still_pending: list[_PendingPlacement] = []
    spend_by_baseline: dict[int, int] = {}
    for pending in _build_gates.pending_placements:
        if wood_now == pending.wood_before and pending.settles_left > 0:
            pending.settles_left -= 1
            still_pending.append(pending)
            continue
        spent = spend_by_baseline.get(pending.wood_before, 0)
        income = _expected_income(pending.noted_at_snapshot)
        purchased = (
            wood_now - income
            <= pending.wood_before - spent - pending.wood_cost + _PLACEMENT_INCOME_SLACK
        )
        if purchased:
            spend_by_baseline[pending.wood_before] = spent + pending.wood_cost
            record_confirmed_buildings([pending.building_class])
            log.info(
                "build_purchase_confirmed",
                building=pending.building_class,
                wood_before=pending.wood_before,
                wood_now=wood_now,
                cost=pending.wood_cost,
                income_estimate=round(income, 1),
            )
        else:
            _note_missing_settlement(pending.building_class)
            log.warning(
                "build_purchase_missing",
                building=pending.building_class,
                wood_before=pending.wood_before,
                wood_now=wood_now,
                cost=pending.wood_cost,
                income_estimate=round(income, 1),
            )
    _build_gates.pending_placements = still_pending


def _clear_missing_streak(building_class: str) -> None:
    """A real purchase proves the build path works — lift any suppression."""
    _build_gates.missing_streaks.pop(building_class, None)
    _build_gates.suppressed_until.pop(building_class, None)


def _note_missing_settlement(building_class: str) -> None:
    """Count a vanished placement; suppress the class after a streak (T-530).

    Run 9 (F-37): 32 consecutive missing farm settlements were retried blindly
    — each one buying an unintended outpost. A streak means something is
    systematically wrong (phantom prerequisite, blocked ground), so stop
    paying for retries and force a pause the LLM can reason about.
    """
    streak = _build_gates.missing_streaks.get(building_class, 0) + 1
    _build_gates.missing_streaks[building_class] = streak
    if streak >= _MISSING_STREAK_LIMIT:
        until = _build_gates.snapshot_count + _MISSING_SUPPRESS_SNAPSHOTS
        _build_gates.suppressed_until[building_class] = until
        log.warning(
            "build_suppressed",
            building=building_class,
            missing_streak=streak,
            until_snapshot=until,
        )


def record_building_sightings(classes: Iterable[str]) -> None:
    """Count one detection frame's building sightings — informational ONLY.

    Sightings never enter buildings_confirmed: a persistent misdetection beats
    any frame-count threshold (run 9, F-36 — a phantom mill unlocked 14
    outposts). Purchase-grade evidence goes through record_confirmed_buildings.
    """
    for cls in set(classes) & _GATE_BUILDING_CLASSES:
        _build_gates.building_sightings[cls] = _build_gates.building_sightings.get(cls, 0) + 1


def record_confirmed_buildings(classes: Iterable[str]) -> None:
    """Remember gate-relevant building classes the agent PROVABLY owns —
    a wood-delta-confirmed purchase or a visually verified placement. The only
    writers of buildings_confirmed (detection sightings stay informational).
    Proof the build path works also lifts any T-530 suppression: run 13
    (F-45) kept a class suppressed after it was verified standing because
    only the wood-delta path cleared the streak."""
    proven = set(classes) & _GATE_BUILDING_CLASSES
    _build_gates.buildings_confirmed.update(proven)
    for cls in proven:
        _clear_missing_streak(cls)


def confirmed_buildings() -> frozenset[str]:
    """Building classes the agent provably built this game (purchase-grade
    evidence) — copied into GameState each turn so the reactive tier can gate
    Feudal prep and the age-up press on the two-building requirement."""
    return frozenset(_build_gates.buildings_confirmed)


def villagers_ordered() -> int:
    """Villagers ordered this game (incl. the 4 starting ones) — copied into
    GameState each turn so the reactive tier gates the queue on ORDERS, not
    the TC-queue-lagged HUD population (run 11, F-38)."""
    return _build_gates.villagers_ordered


def villager_queue_rejection() -> str | None:
    """Reason a villager can't be queued right now (logged), or None.

    The age-keyed order target caps TOTAL orders — the HUD population lags by
    the TC queue depth, so it must never be the brake. An age past the target
    map has no cap. The food gate keeps a press that would silently no-op
    in-game from being counted as an order.
    """
    ordered = _build_gates.villagers_ordered
    target = _VILLAGER_ORDER_TARGET_BY_AGE.get(_build_gates.current_age)
    if target is not None and ordered >= target:
        reason = (
            f"villager target reached ({ordered} ordered, incl. the TC queue) — "
            f"keep villagers busy and bank resources for the "
            f"{_NEXT_AGE[_build_gates.current_age]} instead"
        )
    else:
        food = (_build_gates.resources or {}).get("food")
        if food is None or food >= _VILLAGER_FOOD_COST:
            return None
        reason = f"villager costs {_VILLAGER_FOOD_COST} food, you have {food}"
    log.info("villager_queue_rejected", reason=reason, ordered=ordered)
    return reason


def sighted_buildings() -> frozenset[str]:
    """Building classes detection has seen persistently but nothing proved —
    context-line information only, never gate evidence (F-36)."""
    return frozenset(
        cls
        for cls, frames in _build_gates.building_sightings.items()
        if frames >= _SIGHTING_MIN_FRAMES
    )


def pending_placement_counts() -> Counter[str]:
    """Building classes awaiting wood-delta settlement, by count."""
    return Counter(p.building_class for p in _build_gates.pending_placements)


def reset_build_gates() -> None:
    """Fresh build-gate state (new game / tests)."""
    global _build_gates
    _build_gates = _BuildGates()


def build_rejection(building_key: str, intent: str = "") -> str | None:
    """Reason this build cannot work right now (logged), or None when allowed.

    The single log site for `build_rejected` — every caller (single-shot build
    handler, tool-loop build composite, reassign composite) shapes its own
    failure return but shares this check + log, so the event schema can't drift.
    """
    reason = _rejection_reason(building_key)
    if reason is not None:
        log.info("build_rejected", building_key=building_key, reason=reason, intent=intent)
    return reason


def _rejection_reason(building_key: str) -> str | None:
    """Five gates: suppressed after a missing-settlement streak, unique
    building already standing, house with ample pop-cap headroom (wasted
    wood), missing prerequisite (without a mill the farm key selects the
    OUTPOST — runs 6-7 and 9 built phantom towers this way), and unaffordable
    cost. The reason string is returned to the LLM as the action's failure
    detail so the next turn plans around it instead of re-issuing the same
    doomed build.
    """
    cls = BUILD_KEY_TO_CLASS.get(building_key)
    if cls is None:
        return None
    suppressed_until = _build_gates.suppressed_until.get(cls, 0)
    if _build_gates.snapshot_count < suppressed_until:
        streak = _build_gates.missing_streaks.get(cls, 0)
        return (
            f"{cls} builds suppressed for "
            f"{suppressed_until - _build_gates.snapshot_count} more turns: "
            f"{streak} placements in a row vanished without the wood being spent — "
            "something is systematically wrong (blocked ground, or the "
            "prerequisite isn't really standing)"
        )
    if cls in _UNIQUE_BUILDING_CLASSES:
        if cls in _build_gates.buildings_confirmed:
            return f"{cls} already built — one is enough; spend the wood on farms"
        if any(p.building_class == cls for p in _build_gates.pending_placements):
            return f"{cls} placement already pending wood-delta settlement — don't double-build"
    if cls == "house" and _build_gates.population is not None:
        population, cap = _build_gates.population
        if cap >= _GAME_POP_CAP_LIMIT:
            return f"house skipped: population cap {cap} is already the game maximum"
        headroom = cap - population
        if headroom > _HOUSE_HEADROOM_MAX:
            return (
                f"house skipped: population {population}/{cap} leaves {headroom} headroom "
                f"(> {_HOUSE_HEADROOM_MAX}) — spend the wood on economy buildings instead"
            )
    prereq = _BUILD_PREREQ_CLASS.get(building_key)
    if prereq is not None and prereq not in _build_gates.buildings_confirmed:
        return (
            f"{cls} unavailable: requires a completed {prereq} and none has been "
            f"seen yet — build a {prereq} first"
        )
    cost = _BUILD_WOOD_COST.get(building_key)
    if cost is not None and _build_gates.resources is not None:
        wood = _build_gates.resources.get("wood")
        if wood is not None and wood < cost:
            return f"{cls} unavailable: costs {cost} wood, you have {wood}"
    return None


# ---------------------------------------------------------------------------
# Coordinate resolution
# ---------------------------------------------------------------------------


def _resolve_target_id(target_id: str) -> tuple[int, int] | None:
    """Resolve target_id to (x, y) coordinates from cached entities."""
    for entity in _detected_entities:
        if entity.get("id") == target_id:
            center = entity.get("center")
            if center:
                return (int(center[0]), int(center[1]))
    return None


def _resolve_target_class(target_class: str) -> tuple[int, int] | None:
    """Resolve target_class to (x, y) of first matching entity."""
    for entity in _detected_entities:
        if entity.get("class") == target_class:
            center = entity.get("center")
            if center:
                return (int(center[0]), int(center[1]))
    return None


def _to_int(value: object) -> int:
    """Narrow an action-dict value (typed `object`) to int.

    Action dicts come from LLM output via `dict[str, object]` — the runtime
    values are always int / float / str-of-digits at integer call sites,
    but pyright can't prove that without an explicit narrowing.
    """
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"Expected int-coercible value, got {type(value).__name__}")


def _resolve_coords(action_dict: dict[str, object]) -> tuple[str, tuple[int, int] | None]:
    """Resolve action coordinates from auto_placement, targets, or x/y fields.

    Returns (error_detail, coords). error_detail is non-empty on failure.
    auto_placement resolves NOW — against the entity cache as it is at click
    time, after any camera move earlier in the sequence (run 8, F-33).
    """
    if action_dict.get("auto_placement"):
        return ("", default_build_placement())

    target_id = action_dict.get("target_id")
    if target_id:
        coords = _resolve_target_id(str(target_id))
        if coords is None:
            log.warning("target_id_not_found", target_id=target_id)
            return (f"target_id '{target_id}' not found in detected entities", None)
        return ("", coords)

    target_class = action_dict.get("target_class")
    if target_class:
        coords = _resolve_target_class(str(target_class))
        if coords is None:
            log.warning("target_class_not_found", target_class=target_class)
            return (f"target_class '{target_class}' not found in detected entities", None)
        return ("", coords)

    x, y = action_dict.get("x"), action_dict.get("y")
    if x is not None and y is not None:
        ix, iy = _to_int(x), _to_int(y)
        if ix == 0 and iy == 0:
            log.warning("placeholder_coords_rejected")
            return ("(0, 0) placeholder coordinates rejected", None)
        return ("", (ix, iy))

    return ("no coordinates, target_id, or target_class provided", None)


def can_resolve(action_dict: dict[str, object]) -> bool:
    """Whether a targeted action still resolves against the current entity cache.

    Non-targeted actions (press / scroll / wait / detect) carry no target and
    always pass; targeted ones pass only while their entity is still detected.
    Used by the S6 pipeline to drop committed actions gone stale (RTC).
    """
    if not (action_dict.get("target_id") or action_dict.get("target_class")):
        return True
    error, _coords = _resolve_coords(action_dict)
    return not error


def _translate(x: int, y: int) -> tuple[int, int]:
    """Translate screenshot-relative coords to screen-absolute."""
    return (x + _window_offset[0], y + _window_offset[1])


# ---------------------------------------------------------------------------
# Per-type action handlers
# ---------------------------------------------------------------------------

BUILD_PLACEMENT_KEYWORDS = ("place", "build")
# Hotkeys that re-center the camera — coordinates computed before one of these
# no longer point at the same terrain (run 8, F-33).
CAMERA_KEYS: frozenset[str] = frozenset({"h", ".", ","})
STALE_COORDS_DETAIL = (
    "raw x/y coordinates go stale once the camera moves (a '.'/'h'/',' press "
    "re-centers the view) — use target_class or target_id instead"
)
# Local retry offsets sprayed around an already-open anchor — systematic compass
# points (not random) so coverage is deterministic. The anchor itself is now
# chosen on empty ground (see default_build_placement), so these only need to
# escape small local blockage rather than the whole base.
BUILD_RETRY_RADIUS = 130
BUILD_RETRY_ATTEMPTS = 6
BUILD_SETTLE_DELAY = 0.15
BUILD_RETRY_DELAY = 0.1
RESCAN_SETTLE_DELAY = 0.3
DEFAULT_WAIT_MS = 100

# Ring geometry for picking open build ground around the town centre. The base
# clusters on the TC, so the emptiest ring point is almost always valid ground.
BUILD_RING_RADII: tuple[int, ...] = (280, 400, 520)
BUILD_RING_DIRECTIONS: int = 8
BUILD_CLUTTER_RADIUS: int = 160  # entities within this of a candidate = clutter

# Play-area margins (screenshot px). The HUD occupies the top resource bar and the
# bottom command panel; those regions have no entities, so an emptiness score would
# wrongly rank them as "open" — exclude them from build candidates and gather clicks.
UI_MARGIN_TOP: int = 160
UI_MARGIN_BOTTOM: int = 240
UI_MARGIN_SIDE: int = 40

# Economic build-menu key → detected building class, used to verify a placement
# actually landed (the class appears in the entity cache after a rescan).
BUILD_KEY_TO_CLASS: dict[str, str] = {
    "q": "house",
    "w": "mill",
    "e": "mining_camp",
    "r": "lumber_camp",
    "a": "farm",
    "s": "blacksmith",
    "t": "dock",
}

# Building classes that can serve as build-gate evidence (see
# record_confirmed_buildings) — the econ buildings the agent itself can place.
_GATE_BUILDING_CLASSES: frozenset[str] = frozenset(BUILD_KEY_TO_CLASS.values())

# Wood cost per econ-menu entry (all seven are wood-only). Literals on purpose:
# packages/data's aoe2.db holds the full cost table, but the build gate must not
# depend on a DB handle, and these seven haven't changed in years.
_BUILD_WOOD_COST: dict[str, int] = {
    "q": 25,  # house
    "w": 100,  # mill
    "e": 100,  # mining camp
    "r": 100,  # lumber camp
    "a": 60,  # farm
    "s": 150,  # blacksmith
    "t": 150,  # dock
}

# Menu entries that only exist once a prerequisite building is COMPLETED.
# CRITICAL (user-observed, runs 6-7): without a mill the econ menu re-flows and
# the `A` slot is the OUTPOST — pressing it doesn't no-op, it BUILDS A TOWER.
# This gate is therefore a safety gate, not just an efficiency gate.
_BUILD_PREREQ_CLASS: dict[str, str] = {"a": "mill"}  # farm needs a mill

# One of each is enough for this bot: a second mill/lumber camp is wasted wood
# (run 3 attempted a duplicate mill; run 6's Feudal plan re-emits the lumber
# camp build every turn and relies on this gate to stop once one stands —
# confirmed OR pending, so the settlement lag can't slip a double through).
_UNIQUE_BUILDING_CLASSES: frozenset[str] = frozenset({"mill", "lumber_camp"})

# Module-level cumulative retry telemetry (resets per process / per game).
# Surfaced via build_placement_retry log lines so the user can grep
# `total_count`/`total_seconds` to see how much turn budget got eaten by
# failed placements.
_build_retry_total_seconds: float = 0.0
_build_retry_count: int = 0

# Fallback screen size (retina capture) when the window rect is unavailable.
_DEFAULT_SCREEN: tuple[int, int] = (3024, 1672)
# Any real game window is far larger than this; values below it mean the rect is
# bogus (e.g. the MagicMock pyautogui/pygetwindow shim under headless CI, whose
# int() coerces to 1 rather than raising).
_MIN_WINDOW_DIM: int = 320


def _window_size() -> tuple[int, int]:
    """(width, height) of the game window, or ``_DEFAULT_SCREEN`` when unavailable.

    Defensive against ``get_game_window_rect`` returning ``None`` or a malformed
    rect whose dimensions are absent, non-numeric, or implausibly small.
    """
    rect = get_game_window_rect()
    if rect is not None:
        try:
            width, height = int(rect[2]), int(rect[3])
            if width >= _MIN_WINDOW_DIM and height >= _MIN_WINDOW_DIM:
                return width, height
        except (TypeError, ValueError, IndexError):
            pass
    return _DEFAULT_SCREEN


def _play_area_bounds() -> tuple[int, int, int, int]:
    """(min_x, min_y, max_x, max_y) of the on-map play area, excluding the HUD."""
    width, height = _window_size()
    return (
        UI_MARGIN_SIDE,
        UI_MARGIN_TOP,
        width - UI_MARGIN_SIDE,
        height - UI_MARGIN_BOTTOM,
    )


def _in_play_area(x: int, y: int) -> bool:
    """Whether (x, y) falls on the game map rather than the HUD margins.

    Detections in the top resource bar / bottom command panel / screen edges are
    almost always false positives; right-clicking them sends a villager off into a
    corner instead of onto a resource.
    """
    min_x, min_y, max_x, max_y = _play_area_bounds()
    return min_x <= x <= max_x and min_y <= y <= max_y


def _clutter_score(point: tuple[int, int]) -> int:
    """Number of detected entities within BUILD_CLUTTER_RADIUS of `point`.

    Lower = emptier ground = more likely a valid building spot.
    """
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(
        1
        for entity in _detected_entities
        if (center := entity.get("center")) and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
    )


def _open_ground_candidates(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """Ring points around `anchor` that lie in the play area, emptiest-first."""
    ax, ay = anchor
    min_x, min_y, max_x, max_y = _play_area_bounds()
    candidates: list[tuple[int, int]] = []
    for radius in BUILD_RING_RADII:
        for i in range(BUILD_RING_DIRECTIONS):
            angle = 2.0 * math.pi * i / BUILD_RING_DIRECTIONS
            cx = int(ax + radius * math.cos(angle))
            cy = int(ay + radius * math.sin(angle))
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                candidates.append((cx, cy))
    candidates.sort(key=_clutter_score)
    return candidates


def default_build_placement() -> tuple[int, int]:
    """Screenshot-relative point to start a building placement when the model
    gave no x,y — the executor picks *where*, since the text-only model can't see
    open ground.

    Anchors on the detected town centre but returns the emptiest ring point *around*
    it, never the TC tile itself (clicking on the TC always fails). Falls back to the
    window centre's open ring, then a fixed point, when no TC is detected.
    """
    anchor = _resolve_target_class("town_center")
    if anchor is None:
        width, height = _window_size()
        anchor = (width // 2, height // 2)
    candidates = _open_ground_candidates(anchor)
    return candidates[0] if candidates else anchor


def build_menu_steps(
    building_key: str,
    intent: str,
    *,
    menu_intent: str = "Open economic build menu",
) -> list[dict[str, object]]:
    """Menu → building → place → select-TC sequence, for an ALREADY-selected villager.

    The tail every build shares: open the economic build menu (q), pick the
    building, click the placement (with `building_key` attached so `_handle_click`
    verifies the structure landed), then select the TC so no menu is left open
    to re-map later keystrokes. The placement is resolved AT CLICK TIME
    (`auto_placement`) so a camera move earlier in the sequence can't strand it
    on stale coordinates (run 8, F-33: the mill rose wherever the idle villager
    stood). `build_steps` prepends the idle-villager select;
    `executor_provider.ExecutorProvider._execute_reassign_villager` prepends its own
    worker-click instead — the menu/place sequence lives in exactly one place.
    """
    return [
        {"type": "press", "key": "q", "intent": menu_intent},
        {"type": "press", "key": building_key, "intent": f"Select building ({intent})"},
        {
            "type": "click",
            "auto_placement": True,
            "building_key": building_key,  # lets _handle_click verify the placement landed
            "intent": f"Place building ({intent})",
        },
        # Always leave the UI in a clean state: a menu left open re-maps later
        # keystrokes (runs 6-7 built phantom outposts through leaked menus).
        # Selecting the TC clears menu/ghost by switching selection; `escape`
        # here OPENED the game menu whenever nothing was left to cancel and
        # paused the game (run 8, F-32).
        {"type": "press", "key": "h", "intent": f"Select TC to clear build UI ({intent})"},
    ]


def build_steps(building_key: str, intent: str) -> list[dict[str, object]]:
    """Press/click sequence for a build: select idle villager → open the economic
    build menu → pick the building → place it.

    Shared by the single-shot build handler (`_handle_build`), the tool-loop
    build composite (`executor_provider.ExecutorProvider._execute_build`), and the housed
    fallback, so the steps live in exactly one place. The `.` select re-centers
    the camera on the villager, so it rescans and the placement resolves after
    the jump (F-33).
    """
    return [
        {"type": "press", "key": ".", "rescan": True, "intent": f"Select idle villager ({intent})"},
        *build_menu_steps(building_key, intent),
    ]


async def _handle_click(action_dict: dict[str, object], intent: str) -> ActionResult:
    fail_detail, coords = _resolve_coords(action_dict)
    if coords is None:
        log.warning("click_no_coords", action=action_dict)
        return ActionResult(False, fail_detail)

    x, y = coords
    screen_x, screen_y = _translate(x, y)
    pyautogui.click(screen_x, screen_y)
    log.info(
        "click",
        x=x,
        y=y,
        screen_x=screen_x,
        screen_y=screen_y,
        target_id=action_dict.get("target_id", ""),
        intent=intent,
    )

    if any(word in intent.lower() for word in BUILD_PLACEMENT_KEYWORDS):
        return await _finish_build_placement(action_dict, (x, y), (screen_x, screen_y))
    return ActionResult(True, "ok")


async def _finish_build_placement(
    action_dict: dict[str, object],
    point: tuple[int, int],
    screen_point: tuple[int, int],
) -> ActionResult:
    """Retry-spray a just-clicked building placement, then verify it landed.

    The anchor is already chosen on open ground (default_build_placement), so
    spray a few deterministic compass offsets to escape small local blockage,
    then right-click to cancel the leftover ghost. A build placement consumes
    at the first valid tile, so extra clicks are inert ground clicks. If the
    action carries a building_key, check whether the building appeared.
    """
    global _build_retry_total_seconds, _build_retry_count
    x, y = point
    screen_x, screen_y = screen_point
    retry_start = time.monotonic()
    await asyncio.sleep(BUILD_SETTLE_DELAY)
    offsets = _compass_offsets(BUILD_RETRY_RADIUS, BUILD_RETRY_ATTEMPTS)
    for dx, dy in offsets:
        pyautogui.click(screen_x + dx, screen_y + dy)
        await asyncio.sleep(BUILD_RETRY_DELAY)
    # Cancel any remaining ghost — right-click on the original spot.
    pyautogui.rightClick(screen_x, screen_y)
    elapsed = time.monotonic() - retry_start
    _build_retry_total_seconds += elapsed
    _build_retry_count += 1
    log.debug(
        "build_placement_retry",
        x=x,
        y=y,
        offsets=offsets,
        elapsed_s=round(elapsed, 3),
        total_count=_build_retry_count,
        total_seconds=round(_build_retry_total_seconds, 1),
    )
    building_key = action_dict.get("building_key")
    if isinstance(building_key, str):
        landed = await _verify_build_placement(building_key, point)
        if landed is True:
            # The building is real — usable as prerequisite evidence.
            record_confirmed_buildings([BUILD_KEY_TO_CLASS[building_key]])
        elif landed is False:
            # NOT reported as failure: the model can't see foundations, and a
            # false "failed" makes the LLM rebuild what already exists (the
            # run-2 duplicate mill). The wood spend settles it next snapshot.
            _note_pending_placement(building_key)
            return ActionResult(
                True,
                "placement not visually confirmed (foundations aren't detectable); "
                "will be settled against the wood spend next turn",
            )
        # None = unverifiable (no rescan callback) — benefit of the doubt.
    return ActionResult(True, "ok")


def _compass_offsets(radius: int, count: int) -> list[tuple[int, int]]:
    """`count` evenly-spaced (dx, dy) offsets at `radius` px — a deterministic spray."""
    return [
        (int(radius * math.cos(a)), int(radius * math.sin(a)))
        for i in range(count)
        for a in (2.0 * math.pi * i / count,)
    ]


def _count_class_near(class_name: str, point: tuple[int, int]) -> int:
    """Detected entities of `class_name` within BUILD_CLUTTER_RADIUS of `point`."""
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(
        1
        for entity in _detected_entities
        if entity.get("class") == class_name
        and (center := entity.get("center"))
        and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
    )


async def _verify_build_placement(building_key: str, point: tuple[int, int]) -> bool | None:
    """Rescan and check whether a NEW building of the expected class appeared near
    `point` — the count must increase, so a pre-existing neighbor (e.g. an old
    farm inside the radius) can't vouch for a new one.

    Returns True on a confirmed appearance, False when the rescan saw no new
    one, None when unverifiable (unknown key / no rescan callback). False is
    NOT proof of failure: foundations aren't detectable by the vision model, so
    the caller settles unconfirmed placements against the HUD wood spend
    instead (see _settle_pending_placements).
    """
    expected = BUILD_KEY_TO_CLASS.get(building_key)
    if expected is None or _rescan_fn is None:
        return None
    before = _count_class_near(expected, point)
    await asyncio.sleep(RESCAN_SETTLE_DELAY)
    await _rescan_fn()
    landed = _count_class_near(expected, point) > before
    if landed:
        log.info("build_placement_verified", building=expected, x=point[0], y=point[1])
    else:
        log.info("build_placement_unconfirmed", building=expected, x=point[0], y=point[1])
    return landed


# Classes that appear as the subject ("Send villager to..."), never the target.
_ACTOR_CLASSES = frozenset({"villager", "town_center"})


def _re_resolve_from_intent(x: int, y: int, intent: str) -> tuple[int, int]:
    """Re-resolve raw coordinates using entity class found in the intent.

    The LLM plans all actions from start-of-turn detection. After camera-moving
    keys (H, .) with rescan, those coordinates become stale. This matches the
    entity class mentioned in the intent against freshly detected entities.
    Skips actor classes (villager, town_center) that appear as the subject.
    """
    intent_lower = intent.lower()
    for entity in _detected_entities:
        cls = entity.get("class", "")
        if cls and cls not in _ACTOR_CLASSES and cls in intent_lower:
            resolved = _resolve_target_class(cls)
            if resolved:
                log.debug(
                    "coords_re_resolved",
                    cls=cls,
                    old_x=x,
                    old_y=y,
                    new_x=resolved[0],
                    new_y=resolved[1],
                )
                return resolved
            break
    return (x, y)


async def _handle_right_click(action_dict: dict[str, object], intent: str) -> ActionResult:
    fail_detail, coords = _resolve_coords(action_dict)
    if coords is None:
        log.warning("right_click_no_coords", action=action_dict)
        return ActionResult(False, fail_detail)

    x, y = coords
    if not action_dict.get("target_id") and not action_dict.get("target_class"):
        x, y = _re_resolve_from_intent(x, y, intent)

    if not _in_play_area(x, y):
        log.warning("right_click_off_map", x=x, y=y, intent=intent)
        return ActionResult(False, f"({x}, {y}) is in the HUD margin, not on the map")

    screen_x, screen_y = _translate(x, y)
    pyautogui.rightClick(screen_x, screen_y)
    log.info(
        "right_click",
        x=x,
        y=y,
        screen_x=screen_x,
        screen_y=screen_y,
        target_id=action_dict.get("target_id", ""),
        intent=intent,
    )
    return ActionResult(True, "ok")


async def _handle_press(action_dict: dict[str, object], intent: str) -> ActionResult:
    key = str(action_dict["key"])
    raw_modifiers = action_dict.get("modifiers", [])
    modifiers: list[str] = list(raw_modifiers) if isinstance(raw_modifiers, list) else []
    if modifiers:
        pyautogui.hotkey(*modifiers, key)
        log.info("press", key=key, modifiers=modifiers, intent=intent)
    else:
        pyautogui.press(key)
        log.info("press", key=key, intent=intent)

    # Rescan: take fresh screenshot + detection after camera-moving keys
    if action_dict.get("rescan") and _rescan_fn:
        await asyncio.sleep(RESCAN_SETTLE_DELAY)
        await _rescan_fn()
        log.info("rescan_after_press", key=key)

    return ActionResult(True, "ok")


async def _handle_drag(action_dict: dict[str, object], intent: str) -> ActionResult:
    sx = _to_int(action_dict["start_x"])
    sy = _to_int(action_dict["start_y"])
    ex = _to_int(action_dict["end_x"])
    ey = _to_int(action_dict["end_y"])
    screen_sx, screen_sy = _translate(sx, sy)
    screen_ex, screen_ey = _translate(ex, ey)
    pyautogui.moveTo(screen_sx, screen_sy)
    pyautogui.drag(screen_ex - screen_sx, screen_ey - screen_sy, duration=0.2)
    log.info("drag", start_x=sx, start_y=sy, end_x=ex, end_y=ey, intent=intent)
    return ActionResult(True, "ok")


async def _handle_scroll(action_dict: dict[str, object], intent: str) -> ActionResult:
    clicks = _to_int(action_dict["clicks"])
    x, y = action_dict.get("x"), action_dict.get("y")
    if x is not None and y is not None:
        screen_x, screen_y = _translate(_to_int(x), _to_int(y))
        pyautogui.scroll(clicks, x=screen_x, y=screen_y)
    else:
        pyautogui.scroll(clicks)
    log.info("scroll", clicks=clicks, intent=intent)
    return ActionResult(True, "ok")


async def _handle_detect(_action_dict: dict[str, object], intent: str) -> ActionResult:
    if _rescan_full_fn:
        await _rescan_full_fn()
        log.info("full_detection", intent=intent)
        return ActionResult(True, "ok")
    log.warning("full_detection_unavailable")
    return ActionResult(False, "full detection not available")


async def _handle_wait(action_dict: dict[str, object], intent: str) -> ActionResult:
    ms = _to_int(action_dict.get("ms", DEFAULT_WAIT_MS))
    await asyncio.sleep(ms / 1000)
    log.info("wait", ms=ms, intent=intent)
    return ActionResult(True, "ok")


async def _handle_build(action_dict: dict[str, object], intent: str) -> ActionResult:
    """Build a structure, auto-placed near the Town Center (coordinate-free).

    Runs the shared `build_steps` sequence so the fast single-shot path can build
    too — the executor picks placement since the text-only model can't see open
    ground. The "place" intent triggers `_handle_click`'s blocked-terrain retry.
    """
    key = action_dict.get("building_key")
    if not isinstance(key, str) or not key:
        return ActionResult(False, "build: missing building_key")
    rejection = build_rejection(key, intent)
    if rejection is not None:
        return ActionResult(False, rejection)
    for step in build_steps(key, intent):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"build failed at: {step.get('intent', '')}")
    return ActionResult(True, f"built ({intent})")


async def _handle_queue_villager(action_dict: dict[str, object], intent: str) -> ActionResult:
    """Queue one villager at the TC, through the order ledger (T-531).

    Every queue path (reactive, LLM composite, fallback) funnels here so the
    order target and food gate can't be bypassed by raw h+q presses — the
    invisible-queue overshoot that delivered 40 villagers in run 11 (F-38).
    """
    rejection = villager_queue_rejection()
    if rejection is not None:
        return ActionResult(False, rejection)
    steps: list[dict[str, object]] = [
        {"type": "press", "key": "h", "intent": f"Select TC ({intent})"},
        {"type": "press", "key": "q", "intent": f"Queue villager ({intent})"},
    ]
    for step in steps:
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"queue_villager failed at: {step.get('intent', '')}")
    _build_gates.villagers_ordered += 1
    log.info("villager_ordered", total=_build_gates.villagers_ordered, intent=intent)
    return ActionResult(True, f"villager queued ({_build_gates.villagers_ordered} ordered)")


# Dispatch table: action type -> handler
_ACTION_HANDLERS: dict[
    str,
    Callable[[dict[str, object], str], Awaitable[ActionResult]],
] = {
    "click": _handle_click,
    "right_click": _handle_right_click,
    "press": _handle_press,
    "build": _handle_build,
    "queue_villager": _handle_queue_villager,
    "drag": _handle_drag,
    "scroll": _handle_scroll,
    "detect": _handle_detect,
    "wait": _handle_wait,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_action(action: dict[str, object] | Action) -> ActionResult:
    """Execute a single action from LLM output."""
    # Normalize to dict — isinstance(BaseModel) narrows the type for pyright,
    # which `hasattr` does not.
    if isinstance(action, BaseModel):
        # pyright 1.1.409 fails to narrow `dict[str, object] | Action` to the
        # BaseModel branch here when Action is an Annotated discriminated union.
        action_dict = cast(
            "dict[str, object]",
            action.model_dump(),  # pyright: ignore[reportAttributeAccessIssue]
        )
    else:
        validated = validate_action(action)
        if not validated:
            log.warning("invalid_action", action=action)
            return ActionResult(False, "invalid action format")
        action_dict = cast("dict[str, object]", validated.model_dump())

    action_type_raw = action_dict.get("type", "")
    intent_raw = action_dict.get("intent", "")
    action_type = action_type_raw if isinstance(action_type_raw, str) else ""
    intent = intent_raw if isinstance(intent_raw, str) else ""

    handler = _ACTION_HANDLERS.get(action_type)
    if not handler:
        log.warning("unknown_action", action_type=action_type, action=action_dict)
        return ActionResult(False, f"unknown action type '{action_type}'")

    try:
        # Refresh window offset before each action
        global _window_offset
        rect = get_game_window_rect()
        if rect:
            _window_offset = (rect[0], rect[1])

        result = await handler(action_dict, intent)

        # Move cursor to window center to prevent AoE2 edge-scrolling.
        # Leaving the cursor near screen edges causes camera drift during delays.
        if rect:
            pyautogui.moveTo(rect[0] + 1512, rect[1] + 836)

        # Small delay between actions for stability
        await asyncio.sleep(config.action_delay)
        return result

    except KeyError as e:
        log.error("missing_action_param", action=action_dict, missing=str(e))
        return ActionResult(False, f"missing parameter: {e}")
    except Exception as e:
        log.error("action_failed", action=action_dict, error=str(e))
        return ActionResult(False, f"execution error: {e}")


def _as_dict(action: dict[str, object] | Action) -> dict[str, object]:
    """Plain-dict view of an action for inspection (models are dumped)."""
    if isinstance(action, BaseModel):
        return cast("dict[str, object]", action.model_dump())
    return action


def _moves_camera(action_dict: dict[str, object]) -> bool:
    return action_dict.get("type") == "press" and (
        bool(action_dict.get("rescan")) or str(action_dict.get("key", "")).lower() in CAMERA_KEYS
    )


def _uses_raw_coords_only(action_dict: dict[str, object]) -> bool:
    """A click resolved purely from literal x/y — the form camera moves break."""
    return (
        action_dict.get("type") in ("click", "right_click")
        and action_dict.get("x") is not None
        and not action_dict.get("target_id")
        and not action_dict.get("target_class")
        and not action_dict.get("auto_placement")
    )


async def execute_actions(actions: Sequence[dict[str, object] | Action]) -> list[ActionResult]:
    """Execute a list of actions sequentially.

    A raw-coordinate click after a camera-moving press in the same batch is
    refused instead of executed: its x/y were computed from the pre-move frame
    and land on arbitrary terrain (run 8, F-33 — villagers walked to nothing).
    The failure detail teaches the LLM to name targets instead.
    """
    if not ensure_game_focused():
        log.warning("could_not_focus_before_actions")
        await asyncio.sleep(0.5)
        ensure_game_focused()

    results: list[ActionResult] = []
    camera_moved = False
    for action in actions:
        preview = _as_dict(action)
        if camera_moved and _uses_raw_coords_only(preview):
            log.warning("stale_coords_rejected", intent=preview.get("intent", ""))
            results.append(ActionResult(False, STALE_COORDS_DETAIL))
            continue
        results.append(await execute_action(action))
        camera_moved = camera_moved or _moves_camera(preview)
    return results
