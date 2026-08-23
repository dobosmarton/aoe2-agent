"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import pyautogui
import structlog
from pydantic import BaseModel

from .config import config
from .entity_utils import CLASSES_BY_KIND, nearest_center_of_classes
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.stdlib.get_logger()


def _now() -> float:
    """Monotonic seconds; one seam, so a test can advance the clock. Build-gate
    deadlines are wall clock because perception has its own cadence (Phase 3)."""
    return time.monotonic()


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
# Stale-OCR grace: how long a pending placement waits for the wood reading to
# move before it is settled anyway. 30 s is what the old 3-snapshot grace bought
# at the measured 9.6 s turn (run 2026_08_22_2).
_PLACEMENT_SETTLE_SECONDS = 30.0
# A house settles on the population cap, not the wood delta: 25 wood against a
# 20-wood slack leaves a 5-wood margin on an ESTIMATED income. Run 2026_08_21_2
# built 6 houses and the wood test called 9 confirmed and 21 missing.
_HOUSE_CLASS = "house"
_HOUSE_CAP_STEP = 5  # cap gained per completed house
# Houses need more grace than the wood path: a house takes ~25 s to CONSTRUCT
# before the cap moves, and OCR lag sits on top of that.
_HOUSE_SETTLE_SECONDS = 50.0
# Distinct detection frames before a building class is REPORTED as sighted
# (context line only — sightings never gate builds: run 9, F-36, a persistent
# phantom mill beat any count threshold and 14 outposts got built through the
# unlocked farm slot).
_SIGHTING_MIN_FRAMES = 3
# Build-menu key → detected building class, used to verify a placement actually
# landed (the class appears in the entity cache after a rescan). One map per
# menu, because the same key means different things: econ `w` is the Mill,
# military `w` is the Archery Range.
#
# Tower, wall and castle are deliberately absent from the V menu. core.md warns
# that a tower is stolen economy, and the outpost slot has cost two runs.
ECON_MENU = "q"
MILITARY_MENU = "w"
ADVANCED_MENU = "v"

_MENU_BUILDINGS: dict[str, dict[str, str]] = {
    ECON_MENU: {
        "q": "house",
        "w": "mill",
        "e": "mining_camp",
        "r": "lumber_camp",
        "a": "farm",
        "s": "blacksmith",
        "t": "dock",
    },
    MILITARY_MENU: {
        "q": "barracks",
        "w": "archery_range",
        "e": "stable",
    },
    ADVANCED_MENU: {
        "d": "market",
    },
}

_MENU_NAMES: dict[str, str] = {
    ECON_MENU: "Open economic build menu",
    MILITARY_MENU: "Open military build menu",
    ADVANCED_MENU: "Open advanced build menu",
}


def building_class(menu: str, key: str) -> str | None:
    """The class one menu key places, or None if the menu has no such entry."""
    return _MENU_BUILDINGS.get(menu, {}).get(key)


# The Castle Age needs two buildings FROM the Feudal Age standing; houses, mills
# and camps are Dark Age and do not count. Run 2026_08_21_2 built none of these
# and the age-up stayed greyed out for 13 minutes.
FEUDAL_PREREQ_CLASSES: frozenset[str] = frozenset(
    {"barracks", "archery_range", "stable", "blacksmith", "market"}
)
CASTLE_PREREQ_COUNT = 2


# ---------------------------------------------------------------------------
# Technologies — the research counterpart of the build menus
# ---------------------------------------------------------------------------

# A research is confirmed when the cost resource falls by at least this fraction
# of its price. A fraction, not an exact match, because the HUD reading lags and
# the economy keeps earning; half is wide enough to survive that and still tell
# an 800-food age-up apart from a 50-food villager.
_RESEARCH_CONFIRM_FRACTION = 0.5
# How long a pending research waits for the HUD to move before it is judged.
_RESEARCH_SETTLE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Tech:
    """One researchable item: where to go, which key, what it costs.

    `research_key` is the panel SLOT under the grid layout, because AoE2:DE
    assigns no default hotkey to an upgrade. A wrong slot is no longer silent —
    the settlement reports it once instead of letting it be retried blind.
    """

    goto_key: str
    research_key: str
    goto_modifiers: tuple[str, ...] = ()
    # Building the goto key selects; empty means the Town Center, which always
    # stands. With none standing the goto selects nothing and the research key
    # hits the previous selection — that lost all 5 gold_mining attempts.
    requires: str = ""
    food: int = 0
    gold: int = 0
    wood: int = 0


# Every key verified against the game's own hotkey screen, 2026-08-22. The goto
# keys are its Cycle Commands; the research keys its per-building groups.
_TECHS: dict[str, Tech] = {
    "castle_age": Tech(goto_key="h", research_key="z", food=800, gold=200),
    "loom": Tech(goto_key="h", research_key="a", gold=50),
    "wheelbarrow": Tech(goto_key="h", research_key="s", food=175, wood=50),
    "horse_collar": Tech(
        goto_key="i",
        goto_modifiers=("ctrl",),
        research_key="q",
        requires="mill",
        food=75,
        wood=75,
    ),
    "double_bit_axe": Tech(
        goto_key="z",
        goto_modifiers=("ctrl",),
        research_key="q",
        requires="lumber_camp",
        food=100,
        wood=50,
    ),
    "gold_mining": Tech(
        goto_key="g",
        goto_modifiers=("ctrl",),
        research_key="q",
        requires="mining_camp",
        food=100,
        wood=75,
    ),
}


@dataclass
class _PendingResearch:
    """A research awaiting confirmation from the HUD resource drop."""

    name: str
    tech: Tech
    before: dict[str, int]
    # Monotonic instant after which an undecided reading is judged anyway.
    settle_deadline: float = 0.0


# Circuit breaker (T-530): consecutive missing settlements for one building
# class before its builds are suppressed, and how long the suppression lasts.
# Run 9: 32 identical farm attempts each burned resources.
_MISSING_STREAK_LIMIT = 3
_MISSING_SUPPRESS_SECONDS = 50.0
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


# How `_select_villager_step` picked the villager that builds.
SelectionMode = Literal["click", "idle_press", "unknown"]

# What one HUD snapshot says about a pending placement. "undecided" is not a
# miss: the reading cannot answer yet, so the placement waits for its deadline.
Verdict = Literal["confirmed", "missing", "undecided"]


# A placement whose foundation wasn't visually confirmed, awaiting settlement
# against the HUD wood spend (2026-07-11 run 2, F-11: YOLO can't see foundations,
# so a fresh rescan reports almost every REAL placement as failed — that false
# negative caused a duplicate mill).
@dataclass
class _PendingPlacement:
    building_class: str
    wood_cost: int
    wood_before: int  # wood per the HUD snapshot when the placement was made
    noted_at_snapshot: int  # snapshot_count the wood_before reading belongs to
    cap_before: int = 0  # population cap at the same snapshot — the house signal
    # How the villager was selected and where the click landed, so a missing
    # settlement names its own cause.
    selected_by: SelectionMode = "unknown"
    point: tuple[int, int] = (0, 0)
    # Monotonic instant after which an undecided reading is judged anyway.
    settle_deadline: float = 0.0

    @property
    def is_house(self) -> bool:
        return self.building_class == _HOUSE_CLASS


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
    pending_placements: builds awaiting wood-delta settlement; pending_research
    is its technology counterpart, settled on the food/gold drop.
    """

    population: tuple[int, int] | None = None
    resources: dict[str, int] | None = None
    # This iteration's idle-villager reading, and how the last build acted on
    # it — see _select_villager_step, which owns both.
    idle_present: bool | None = None
    selected_by: SelectionMode = "unknown"
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
    # the monotonic instant until which a repeatedly-missing class stays blocked.
    missing_streaks: dict[str, int] = field(default_factory=dict)
    suppressed_until: dict[str, float] = field(default_factory=dict)
    # The research counterparts: awaiting settlement, proven paid for, and the
    # instant until which a proven miss stays blocked.
    pending_research: list[_PendingResearch] = field(default_factory=list)
    researched: set[str] = field(default_factory=set)
    research_blocked_until: dict[str, float] = field(default_factory=dict)
    snapshot_count: int = 0
    # Villagers ordered so far (T-531) — self-generated ground truth that
    # leads the delivered HUD population by the TC queue depth.
    villagers_ordered: int = _STARTING_VILLAGERS
    # Validated age from GameState (strategist OCR), synced once per turn —
    # selects the villager order target and the rejection message (T-538).
    current_age: str = "Dark Age"


_build_gates = _BuildGates()


def observe_hud(
    population: int,
    population_cap: int,
    resources: Mapping[str, int],
    *,
    idle_present: bool | None = None,
) -> None:
    """Feed this turn's HUD reading into the build gates.

    Not just a cache write: the fresh wood value first updates the gather-income
    estimate, then SETTLES pending placements (confirming purchases / flagging
    missing ones) before the snapshot is replaced — the ordering the wood-delta
    check depends on.
    """
    _build_gates.snapshot_count += 1
    _observe_wood_income(resources.get("wood"))
    # Both readings are passed in, not read off the gates: settlement runs
    # BEFORE the snapshot is replaced, which is the ordering the deltas need.
    _settle_pending_placements(resources.get("wood"), population_cap)
    _settle_pending_research(resources)
    _build_gates.population = (population, population_cap)
    _build_gates.resources = dict(resources)
    _build_gates.idle_present = idle_present


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


def _note_pending_placement(
    building_key: str, *, menu: str = ECON_MENU, point: tuple[int, int] = (0, 0)
) -> None:
    """Queue an unconfirmed placement for wood-delta settlement next snapshot."""
    cls = building_class(menu, building_key)
    cost = _WOOD_COST_BY_CLASS.get(cls or "")
    wood_before = (_build_gates.resources or {}).get("wood")
    if cls is None or cost is None or wood_before is None:
        # No wood baseline to settle against — the placement stays unconfirmed
        # for good, despite the caller's "settled next turn" detail. Say so.
        log.debug("placement_pending_dropped", building_key=building_key)
        return
    _, cap_before = _build_gates.population or (0, 0)
    is_house = cls == _HOUSE_CLASS
    _build_gates.pending_placements.append(
        _PendingPlacement(
            building_class=cls,
            wood_cost=cost,
            wood_before=wood_before,
            noted_at_snapshot=_build_gates.snapshot_count,
            cap_before=cap_before,
            selected_by=_build_gates.selected_by,
            point=point,
            settle_deadline=_now()
            + (_HOUSE_SETTLE_SECONDS if is_house else _PLACEMENT_SETTLE_SECONDS),
        )
    )


def _settle_pending_placements(wood_now: int | None, cap_now: int) -> None:
    """Confirm or drop pending placements using the game's own ledger — the HUD.

    A placement that consumed its cost DID succeed regardless of what detection
    saw (the HUD is authoritative; the vision model can't see foundations).
    Houses settle on the population cap and every other class on the wood delta
    — see `_house_verdict` and `_wood_verdict`. Judged FIFO, erring toward
    confirmation: a false "failed" report is what caused the duplicate mill,
    while a false success merely delays the retry by a turn. An "undecided"
    reading waits for the next snapshot, until `settle_deadline` passes.

    A missing wood reading stops the whole pass, houses included: one OCR frame
    supplies both numbers, so an unreadable wood value means an unreliable cap.
    """
    if not _build_gates.pending_placements or wood_now is None:
        return
    still_pending: list[_PendingPlacement] = []
    spend_by_baseline: dict[int, int] = {}
    claimed_cap: dict[int, int] = {}
    for pending in _build_gates.pending_placements:
        verdict = (
            _house_verdict(pending, cap_now, claimed_cap)
            if pending.is_house
            else _wood_verdict(pending, wood_now, spend_by_baseline)
        )
        if verdict == "undecided" and _now() < pending.settle_deadline:
            still_pending.append(pending)
            continue
        evidence = _settlement_evidence(pending, wood_now, cap_now)
        if verdict == "confirmed":
            record_confirmed_buildings([pending.building_class])
            log.info("build_purchase_confirmed", **evidence)
        else:
            _note_missing_settlement(pending.building_class)
            log.warning(
                "build_purchase_missing",
                **evidence,
                selected_by=pending.selected_by,
                x=pending.point[0],
                y=pending.point[1],
            )
    _build_gates.pending_placements = still_pending


def _house_verdict(pending: _PendingPlacement, cap_now: int, claimed: dict[int, int]) -> Verdict:
    """Confirmed once the population cap has risen by a whole house.

    Never "missing": an unmoved cap means the house may still be under
    construction. `claimed` stops one +10 jump from confirming three pending
    houses — the cap analogue of `spend_by_baseline` (run 3, F-17).
    """
    already = claimed.get(pending.cap_before, 0)
    if cap_now - pending.cap_before - already < _HOUSE_CAP_STEP:
        return "undecided"
    claimed[pending.cap_before] = already + _HOUSE_CAP_STEP
    return "confirmed"


def _wood_verdict(
    pending: _PendingPlacement, wood_now: int, spend_by_baseline: dict[int, int]
) -> Verdict:
    """Whether the HUD wood delta covers this placement's cost.

    An unchanged reading is stale OCR, not a miss.

    Estimated gather income is deducted from the delta first — run 13 (F-45)
    gathered +140 wood across a 25-wood house settlement, so the raw delta alone
    judged every real purchase MISSING and the circuit breaker locked out five
    classes. Confirmed spend is deducted per shared baseline, so one wood drop
    confirms at most one pending of a given cost.
    """
    if wood_now == pending.wood_before:
        return "undecided"
    spent = spend_by_baseline.get(pending.wood_before, 0)
    income = _expected_income(pending.noted_at_snapshot)
    budget = pending.wood_before - spent - pending.wood_cost + _PLACEMENT_INCOME_SLACK
    if wood_now - income > budget:
        return "missing"
    spend_by_baseline[pending.wood_before] = spent + pending.wood_cost
    return "confirmed"


def _settlement_evidence(
    pending: _PendingPlacement, wood_now: int, cap_now: int
) -> dict[str, object]:
    """The numbers the settlement judged on, for either log line."""
    if pending.is_house:
        return {
            "building": pending.building_class,
            "cap_before": pending.cap_before,
            "cap_now": cap_now,
        }
    return {
        "building": pending.building_class,
        "wood_before": pending.wood_before,
        "wood_now": wood_now,
        "cost": pending.wood_cost,
        "income_estimate": round(_expected_income(pending.noted_at_snapshot), 1),
    }


def _note_pending_research(name: str, tech: Tech) -> None:
    """Queue a research for HUD settlement next snapshot."""
    before = _build_gates.resources
    if before is None:
        log.debug("research_pending_dropped", tech=name)
        return
    _build_gates.pending_research.append(
        _PendingResearch(
            name=name,
            tech=tech,
            before=dict(before),
            settle_deadline=_now() + _RESEARCH_SETTLE_SECONDS,
        )
    )


def _settle_pending_research(resources: Mapping[str, int]) -> None:
    """Confirm or report each pending research against the HUD resource drop.

    This is the feedback a raw `press` never had: a keystroke always "succeeds",
    so a greyed-out button looked identical to a working one. Run 2026_08_21_2
    pressed the age-up key 10 times over 4 minutes on that blind spot.
    """
    if not _build_gates.pending_research:
        return
    still_pending: list[_PendingResearch] = []
    for pending in _build_gates.pending_research:
        verdict = _research_verdict(pending, resources)
        if verdict == "undecided" and _now() < pending.settle_deadline:
            still_pending.append(pending)
            continue
        if verdict == "confirmed":
            _build_gates.researched.add(pending.name)
            log.info("research_confirmed", tech=pending.name)
        else:
            _build_gates.research_blocked_until[pending.name] = _now() + _MISSING_SUPPRESS_SECONDS
            log.warning(
                "research_missing",
                tech=pending.name,
                retry_in_s=round(_MISSING_SUPPRESS_SECONDS),
                **_research_costs(pending.tech),
            )
    _build_gates.pending_research = still_pending


def _research_verdict(pending: _PendingResearch, resources: Mapping[str, int]) -> Verdict:
    """Confirmed once every cost resource has fallen far enough to have paid.

    ANY unchanged cost reading is stale OCR, not a refusal: a frame where food
    updated but gold did not once reported a paid-for age-up as missing.
    """
    shortfalls = []
    for kind, price in _research_costs(pending.tech).items():
        now = resources.get(kind)
        was = pending.before.get(kind)
        if now is None or was is None or now == was:
            return "undecided"
        shortfalls.append(was - now < price * _RESEARCH_CONFIRM_FRACTION)
    return "missing" if any(shortfalls) else "confirmed"


def _research_costs(tech: Tech) -> dict[str, int]:
    """The non-zero prices of one technology, keyed by resource."""
    return {
        kind: price
        for kind, price in (("food", tech.food), ("gold", tech.gold), ("wood", tech.wood))
        if price
    }


def _is_pop_capped() -> bool:
    """Whether the HUD shows no population headroom. False with no reading yet."""
    if _build_gates.population is None:
        return False
    population, cap = _build_gates.population
    return cap > 0 and population >= cap


def is_pop_capped() -> bool:
    """Whether the HUD says no villager can be queued. False when unread."""
    return _is_pop_capped()


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

    Never suppresses a house while pop-capped: a house is the ONLY way out, so
    the pause becomes a deadlock. Run 2026_08_21_2 sat at 35/35 for the last 10
    minutes of a 25-minute game with houses suppressed 13 times.
    """
    if building_class == _HOUSE_CLASS and _is_pop_capped():
        return
    streak = _build_gates.missing_streaks.get(building_class, 0) + 1
    _build_gates.missing_streaks[building_class] = streak
    if streak >= _MISSING_STREAK_LIMIT:
        _build_gates.suppressed_until[building_class] = _now() + _MISSING_SUPPRESS_SECONDS
        log.warning(
            "build_suppressed",
            building=building_class,
            missing_streak=streak,
            retry_in_s=round(_MISSING_SUPPRESS_SECONDS),
        )


def record_building_sightings(classes: Iterable[str]) -> None:
    """Count one detection frame's building sightings — informational ONLY.

    Sightings never enter buildings_confirmed: a persistent misdetection beats
    any frame-count threshold (run 9, F-36 — a phantom mill unlocked 14
    outposts). Purchase-grade evidence goes through record_confirmed_buildings.
    """
    for cls in set(classes) & GATE_BUILDING_CLASSES:
        _build_gates.building_sightings[cls] = _build_gates.building_sightings.get(cls, 0) + 1


def record_confirmed_buildings(classes: Iterable[str]) -> None:
    """Remember gate-relevant building classes the agent PROVABLY owns —
    a wood-delta-confirmed purchase or a visually verified placement. The only
    writers of buildings_confirmed (detection sightings stay informational).
    Proof the build path works also lifts any T-530 suppression: run 13
    (F-45) kept a class suppressed after it was verified standing because
    only the wood-delta path cleared the streak."""
    proven = set(classes) & GATE_BUILDING_CLASSES
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


def blocked_actions() -> list[str]:
    """Refusals the LLM cannot work out for itself: suppressed builds, blocked
    researches, then finished ones, alphabetical within each group.

    Read off the gate state rather than the rejection helpers, which log — a
    context line must stay side-effect free. Affordability is absent on purpose:
    the resources sit two lines above it.
    """
    now = _now()
    blocked = [
        f"{cls} (suppressed {round(until - now)}s)"
        for cls, until in sorted(_build_gates.suppressed_until.items())
        if now < until
    ]
    blocked += [
        f"{name} (retryable in {round(until - now)}s)"
        for name, until in sorted(_build_gates.research_blocked_until.items())
        if now < until
    ]
    blocked += [f"{name} (already researched)" for name in sorted(_build_gates.researched)]
    return blocked


def pending_placement_counts() -> Counter[str]:
    """Building classes awaiting wood-delta settlement, by count."""
    return Counter(p.building_class for p in _build_gates.pending_placements)


def reset_build_gates() -> None:
    """Fresh build-gate state (new game / tests)."""
    global _build_gates
    _build_gates = _BuildGates()


def build_rejection(building_key: str, intent: str = "", *, menu: str = ECON_MENU) -> str | None:
    """Reason this build cannot work right now (logged), or None when allowed.

    The single log site for `build_rejected` — every caller (single-shot build
    handler, tool-loop build composite, reassign composite) shapes its own
    failure return but shares this check + log, so the event schema can't drift.
    """
    reason = _rejection_reason(building_key, menu)
    if reason is not None:
        log.info("build_rejected", building_key=building_key, reason=reason, intent=intent)
    return reason


def _committed_wood() -> int:
    """Wood owed by placements the HUD has not settled yet.

    The reading refreshes once per turn, so without this a second build sees
    money the first already spent — run 2026_08_22_2 had 125 wood and committed
    200. Self-limiting: a pending placement is judged by its settle deadline.
    """
    return sum(p.wood_cost for p in _build_gates.pending_placements)


def _rejection_reason(building_key: str, menu: str) -> str | None:
    """Five gates: suppressed after a missing-settlement streak, unique
    building already standing, house with ample pop-cap headroom (wasted
    wood), missing prerequisite (without a mill the farm key selects the
    OUTPOST — runs 6-7 and 9 built phantom towers this way), and unaffordable
    cost. The reason string is returned to the LLM as the action's failure
    detail so the next turn plans around it instead of re-issuing the same
    doomed build.
    """
    cls = building_class(menu, building_key)
    if cls is None:
        return None
    suppressed_until = _build_gates.suppressed_until.get(cls, 0.0)
    if _now() < suppressed_until:
        streak = _build_gates.missing_streaks.get(cls, 0)
        return (
            f"{cls} builds suppressed for "
            f"{round(suppressed_until - _now())} more seconds: "
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
    cost = _WOOD_COST_BY_CLASS.get(cls)
    if cost is not None and _build_gates.resources is not None:
        wood = _build_gates.resources.get("wood")
        committed = _committed_wood()
        if wood is not None and wood - committed < cost:
            spare = wood - committed
            if committed:
                return (
                    f"{cls} unavailable: costs {cost} wood and only {spare} is uncommitted "
                    f"({wood} on the HUD, {committed} owed by placements not yet settled)"
                )
            return f"{cls} unavailable: costs {cost} wood, you have {wood}"
    if building_key in _RESOURCE_REQUIRED_KEYS and _resource_anchor(building_key) is None:
        classes = ", ".join(sorted(_BUILD_ANCHOR_CLASSES[building_key]))
        return (
            f"{cls} skipped: no {classes} visible to build against — a drop-off camp "
            "away from its resource carries nothing; wait for the view to show one"
        )
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
        key = str(action_dict.get("building_key", ""))
        placement = default_build_placement(key)
        if placement is None:
            # The camera moved since the pre-flight gate ran, so the resource that
            # authorised this build is no longer in frame. The trailing `h` press
            # in build_menu_steps still clears the open menu.
            return (
                f"no visible resource to anchor the {building_class(ECON_MENU, key) or key} on",
                None,
            )
        return ("", placement)

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
# Ring for a drop-off camp, measured from the RESOURCE. It has to land adjacent,
# so these hug far tighter than the TC ring above (a tile is ~130px at the
# deployment capture size — see BUILD_RETRY_RADIUS).
RESOURCE_RING_RADII: tuple[int, ...] = (150, 210, 270)
BUILD_RING_DIRECTIONS: int = 8
BUILD_CLUTTER_RADIUS: int = 160  # entities within this of a candidate = clutter

# Play-area margins (screenshot px). The HUD occupies the top resource bar and the
# bottom command panel; those regions have no entities, so an emptiness score would
# wrongly rank them as "open" — exclude them from build candidates and gather clicks.
UI_MARGIN_TOP: int = 160
UI_MARGIN_BOTTOM: int = 240
UI_MARGIN_SIDE: int = 40

# Drop-off camps are worthless away from their resource, so they anchor on it
# instead of the town centre. A key absent here keeps the TC anchor.
_BUILD_ANCHOR_CLASSES: dict[str, frozenset[str]] = {
    "r": CLASSES_BY_KIND["wood"],  # lumber camp → tree
    "e": CLASSES_BY_KIND["gold"] | CLASSES_BY_KIND["stone"],  # mining camp → either mine
    "w": frozenset({"berry_bush"}),  # mill → berries
}
# The mill is the only anchored building that still places without its resource:
# it is also the farm unlock and a Feudal prerequisite, and run 12 (F-41) starved
# because only the LLM ever built one.
_ANCHOR_OPTIONAL_KEYS: frozenset[str] = frozenset({"w"})
# Derived, so a newly anchored building waits for its resource by default.
_RESOURCE_REQUIRED_KEYS: frozenset[str] = frozenset(_BUILD_ANCHOR_CLASSES) - _ANCHOR_OPTIONAL_KEYS

# Building classes that can serve as build-gate evidence (see
# record_confirmed_buildings) — every building the agent itself can place.
# Menu-wide, not econ-only: a barracks that cannot become evidence can never
# count toward the Castle Age's two-building requirement.
GATE_BUILDING_CLASSES: frozenset[str] = frozenset(
    cls for menu in _MENU_BUILDINGS.values() for cls in menu.values()
)

# Wood cost per building class (every one of these is wood-only). Literals on
# purpose: packages/data's aoe2.db holds the full cost table, but the build gate
# must not depend on a DB handle, and these costs haven't changed in years.
_WOOD_COST_BY_CLASS: dict[str, int] = {
    "house": 25,
    "farm": 60,
    "mill": 100,
    "mining_camp": 100,
    "lumber_camp": 100,
    "blacksmith": 150,
    "dock": 150,
    "barracks": 175,
    "archery_range": 175,
    "stable": 175,
    "market": 175,
}

# The econ menu's costs by key — the shape the reactive rules' `cost` blocks and
# their drift test read.
_BUILD_WOOD_COST: dict[str, int] = {
    key: _WOOD_COST_BY_CLASS[cls] for key, cls in _MENU_BUILDINGS[ECON_MENU].items()
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

    Lower = emptier ground = more likely a valid building spot. Counting the
    resource itself is deliberate: on a camp's tight ring it makes the emptiest
    candidate the open ground at the forest's edge.
    """
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(
        1
        for entity in _detected_entities
        if (center := entity.get("center")) and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
    )


def _open_ground_candidates(
    anchor: tuple[int, int], radii: tuple[int, ...] = BUILD_RING_RADII
) -> list[tuple[int, int]]:
    """Ring points around `anchor` that lie in the play area, emptiest-first."""
    ax, ay = anchor
    min_x, min_y, max_x, max_y = _play_area_bounds()
    candidates: list[tuple[int, int]] = []
    for radius in radii:
        for i in range(BUILD_RING_DIRECTIONS):
            angle = 2.0 * math.pi * i / BUILD_RING_DIRECTIONS
            cx = int(ax + radius * math.cos(angle))
            cy = int(ay + radius * math.sin(angle))
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                candidates.append((cx, cy))
    candidates.sort(key=_clutter_score)
    return candidates


def _home_anchor() -> tuple[int, int]:
    """The base's screen point: the detected town centre, else the view centre."""
    tc = _resolve_target_class("town_center")
    if tc is not None:
        return tc
    width, height = _window_size()
    return (width // 2, height // 2)


def _resource_anchor(building_key: str) -> tuple[int, int] | None:
    """Center of the resource a drop-off camp should hug, or None if none is visible.

    Nearest to the town centre rather than to the camera, so a camp lands at the
    home forest instead of whichever tree the view happens to show.
    """
    classes = _BUILD_ANCHOR_CLASSES.get(building_key)
    if classes is None:
        return None
    center = nearest_center_of_classes(_detected_entities, classes, _home_anchor())
    return None if center is None else (int(center[0]), int(center[1]))


def default_build_placement(building_key: str) -> tuple[int, int] | None:
    """Screenshot-relative point to start a placement, since the text-only model
    can't see open ground and the schema carries no coordinates.

    A drop-off camp takes the emptiest point on a tight ring around its resource;
    everything else takes one around the town centre, never the TC tile itself
    (clicking on the TC always fails). None means the camp's resource is off
    screen and the caller should skip the turn.
    """
    resource = _resource_anchor(building_key)
    if resource is not None:
        candidates = _open_ground_candidates(resource, RESOURCE_RING_RADII)
        # The fallback puts the camp ON its mine, where nothing can be built —
        # a suspect for run 2026_08_22_1's 12 misses. Logged, not guessed at.
        point = candidates[0] if candidates else resource
        log.debug(
            "anchored_placement",
            building_key=building_key,
            anchor=resource,
            point=point,
            offset=round(math.dist(resource, point)),
            candidates=len(candidates),
        )
        return point
    if building_key in _RESOURCE_REQUIRED_KEYS:
        return None
    anchor = _home_anchor()
    candidates = _open_ground_candidates(anchor)
    return candidates[0] if candidates else anchor


def build_menu_steps(
    building_key: str,
    intent: str,
    *,
    menu: str = ECON_MENU,
    menu_intent: str = "",
) -> list[dict[str, object]]:
    """Menu → building → place → select-TC sequence, for an ALREADY-selected villager.

    The tail every build shares: open a build menu, pick the building, click the
    placement (with `building_key` and `menu` attached so `_handle_click` verifies
    the structure landed), then select the TC so no menu is left open to re-map
    later keystrokes. The placement is resolved AT CLICK TIME (`auto_placement`)
    so a camera move earlier in the sequence can't strand it on stale coordinates
    (run 8, F-33: the mill rose wherever the idle villager stood). `build_steps`
    prepends the idle-villager select;
    `executor_provider.ExecutorProvider._execute_reassign_villager` prepends its own
    worker-click instead — the menu/place sequence lives in exactly one place.
    """
    return [
        {"type": "press", "key": menu, "intent": menu_intent or _MENU_NAMES[menu]},
        {"type": "press", "key": building_key, "intent": f"Select building ({intent})"},
        {
            "type": "click",
            "auto_placement": True,
            # Both, so _handle_click can verify the placement landed.
            "building_key": building_key,
            "menu": menu,
            "intent": f"Place building ({intent})",
        },
        # Always leave the UI in a clean state: a menu left open re-maps later
        # keystrokes (runs 6-7 built phantom outposts through leaked menus).
        # Selecting the TC clears menu/ghost by switching selection; `escape`
        # here OPENED the game menu whenever nothing was left to cancel and
        # paused the game (run 8, F-32).
        {"type": "press", "key": "h", "intent": f"Select TC to clear build UI ({intent})"},
    ]


def research_steps(name: str, intent: str) -> list[dict[str, object]]:
    """Go to the building that researches `name`, then press its panel key.

    Two steps, one place — shared by the single-shot handler and the tool-loop
    composite, exactly as `build_steps` is.
    """
    tech = _TECHS[name]
    return [
        {
            "type": "press",
            "key": tech.goto_key,
            "modifiers": list(tech.goto_modifiers),
            "rescan": True,
            "intent": f"Go to the {name} building ({intent})",
        },
        {"type": "press", "key": tech.research_key, "intent": f"Research {name} ({intent})"},
    ]


def research_rejection(name: str) -> str | None:
    """Reason this research cannot work now (logged), or None when allowed.

    The counterpart of `build_rejection`: the reason reaches the LLM as the
    action's failure detail, so a technology that has already been paid for — or
    one the HUD proved did not take — is not retried blind.
    """
    reason = _research_rejection_reason(name)
    if reason is not None:
        log.info("research_rejected", tech=name, reason=reason)
    return reason


def _research_rejection_reason(name: str) -> str | None:
    """Five gates, in order: unknown name, already paid for, blocked after a
    proven miss, already awaiting settlement, its building not standing — then
    affordability."""
    tech = _TECHS.get(name)
    if tech is None:
        return f"unknown technology {name!r}; known: {', '.join(sorted(_TECHS))}"
    if name in _build_gates.researched:
        return f"{name} is already researched — the HUD showed it paid for"
    blocked_until = _build_gates.research_blocked_until.get(name, 0.0)
    if _now() < blocked_until:
        return (
            f"{name} did not take last time: the cost never left the HUD, so the "
            f"button was greyed out. Satisfy its requirement — retryable in "
            f"{round(blocked_until - _now())} seconds"
        )
    if any(p.name == name for p in _build_gates.pending_research):
        return f"{name} is already awaiting HUD settlement — don't re-press it"
    if tech.requires and tech.requires not in _build_gates.buildings_confirmed:
        return (
            f"{name} is researched at a {tech.requires} and none is confirmed "
            f"standing — build a {tech.requires} first"
        )
    resources = _build_gates.resources
    if resources is None:
        return None
    for kind, price in _research_costs(tech).items():
        have = resources.get(kind)
        if have is not None and have < price:
            return f"{name} unavailable: costs {price} {kind}, you have {have}"
    return None


def _select_villager_step(intent: str) -> dict[str, object]:
    """Select the villager that will build, and record how (`selected_by`).

    "." is preferred: it takes an IDLE villager and re-centers the camera, so
    the placement resolves after the jump (F-33). But "." is a no-op when
    nothing is idle, and every build ends by pressing "h" — the Town Center then
    stays selected and the next "q" queues a villager instead of opening the
    menu. Run 2026_08_21_1 lost 19 of 25 placements that way.
    """
    nothing_is_idle = _build_gates.idle_present is False  # None = no reading yet
    _build_gates.selected_by = "click" if nothing_is_idle else "idle_press"
    if nothing_is_idle:
        return {
            "type": "click",
            "target_class": "villager",
            "intent": f"Select villager ({intent})",
        }
    return {
        "type": "press",
        "key": ".",
        "rescan": True,
        "intent": f"Select idle villager ({intent})",
    }


def build_steps(
    building_key: str, intent: str, *, menu: str = ECON_MENU
) -> list[dict[str, object]]:
    """Press/click sequence for a build: select a villager → open the economic
    build menu → pick the building → place it.

    Shared by the single-shot build handler (`_handle_build`), the tool-loop
    build composite (`executor_provider.ExecutorProvider._execute_build`), and the housed
    fallback, so the steps live in exactly one place.
    """
    return [
        _select_villager_step(intent),
        *build_menu_steps(building_key, intent, menu=menu),
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
    menu = str(action_dict.get("menu") or ECON_MENU)
    if isinstance(building_key, str):
        cls = building_class(menu, building_key)
        landed = await _verify_build_placement(cls, point)
        if landed is True and cls is not None:
            # The building is real — usable as prerequisite evidence.
            record_confirmed_buildings([cls])
        elif landed is False:
            # NOT reported as failure: the model can't see foundations, and a
            # false "failed" makes the LLM rebuild what already exists (the
            # run-2 duplicate mill). The wood spend settles it next snapshot.
            _note_pending_placement(building_key, menu=menu, point=point)
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


async def _verify_build_placement(expected: str | None, point: tuple[int, int]) -> bool | None:
    """Rescan and check whether a NEW building of `expected` class appeared near
    `point` — the count must increase, so a pre-existing neighbor (e.g. an old
    farm inside the radius) can't vouch for a new one.

    Returns True on a confirmed appearance, False when the rescan saw no new
    one, None when unverifiable (unknown class / no rescan callback). False is
    NOT proof of failure: foundations aren't detectable by the vision model, so
    the caller settles unconfirmed placements against the HUD spend instead
    (see _settle_pending_placements).
    """
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
    menu = str(action_dict.get("menu") or ECON_MENU)
    rejection = build_rejection(key, intent, menu=menu)
    if rejection is not None:
        return ActionResult(False, rejection)
    for step in build_steps(key, intent, menu=menu):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"build failed at: {step.get('intent', '')}")
    return ActionResult(True, f"built ({intent})")


async def _handle_research(action_dict: dict[str, object], intent: str) -> ActionResult:
    """Research a technology, then leave it pending HUD settlement.

    Reports success optimistically for the same reason a placement does: the
    press has landed and only the next HUD reading can say whether it paid. A
    refusal the gates already know about comes back as a failure detail, so the
    LLM plans around it instead of re-pressing (run 2026_08_21_2, 10 blind
    age-up presses).
    """
    name = action_dict.get("tech")
    if not isinstance(name, str) or not name:
        return ActionResult(False, "research: missing tech")
    rejection = research_rejection(name)
    if rejection is not None:
        return ActionResult(False, rejection)
    for step in research_steps(name, intent):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"research failed at: {step.get('intent', '')}")
    _note_pending_research(name, _TECHS[name])
    return ActionResult(
        True,
        f"{name} pressed; the HUD spend settles it next turn — do not re-press it",
    )


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
    "research": _handle_research,
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
