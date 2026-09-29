"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import pyautogui
import structlog
from pydantic import BaseModel

from .config import config
from .entity_utils import (
    CLASSES_BY_KIND,
    RESOURCE_KINDS,
    ResourceKind,
    nearest_center_of_classes,
    safe_gather_target,
)
from .models import Action, validate_action
from .policy.candidates import eligible
from .policy.catalog import (
    BUILDINGS,
    BY_ID,
    FEUDAL_BUILDINGS,
    RESEARCH,
    UNITS,
)
from .policy.state import PolicyState
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
_rescan_fn: Callable[[], Awaitable[bool]] | None = None
_rescan_full_fn: Callable[[], Awaitable[bool]] | None = None
_selection_refresh_fn: Callable[[], Awaitable[bool]] | None = None


@dataclass
class ActionResult:
    """Result of executing a single action."""

    success: bool
    detail: str


# ---------------------------------------------------------------------------
# Entity cache management
# ---------------------------------------------------------------------------


def set_rescan_fn(fn: Callable[[], Awaitable[bool]]) -> None:
    """Set the rescan callback for mid-turn screenshot+detection."""
    global _rescan_fn
    _rescan_fn = fn


def set_rescan_full_fn(fn: Callable[[], Awaitable[bool]]) -> None:
    """Set the full detection callback for thorough SAHI scan."""
    global _rescan_full_fn
    _rescan_full_fn = fn


def set_selection_refresh_fn(fn: Callable[[], Awaitable[bool]]) -> None:
    """Set the quick post-input HUD and command-panel refresh callback."""
    global _selection_refresh_fn
    _selection_refresh_fn = fn


def clear_refresh_callbacks() -> None:
    """Release per-game capture callbacks after all loops have stopped."""
    global _rescan_fn, _rescan_full_fn, _selection_refresh_fn
    _rescan_fn = None
    _rescan_full_fn = None
    _selection_refresh_fn = None


async def _selection_refresh() -> bool:
    callback = _selection_refresh_fn or _rescan_fn
    return callback is not None and await callback()


def get_rescan_fn() -> Callable[[], Awaitable[bool]] | None:
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

# The game's population-cap maximum: houses past it add nothing.
_GAME_POP_CAP_LIMIT = 200
# After this deadline, an inconclusive placement is reported uncertain but
# retains its reservation and duplicate guard until evidence arrives.
_PLACEMENT_SETTLE_SECONDS = 30.0
# A completed house is visible in the population cap even if gathering masked
# its 25-wood purchase in the resource HUD.
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
    menu: {key: spec.subject for (group, key), spec in BUILDINGS.items() if group == menu}
    for menu in (ECON_MENU, MILITARY_MENU, ADVANCED_MENU)
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
FEUDAL_PREREQ_CLASSES: frozenset[str] = FEUDAL_BUILDINGS
CASTLE_PREREQ_COUNT = 2


# ---------------------------------------------------------------------------
# Technologies — the research counterpart of the build menus
# ---------------------------------------------------------------------------

# How long a pending research waits for the HUD to move before it is judged.
_RESEARCH_SETTLE_SECONDS = 30.0
_ASSIGNMENT_SETTLE_SECONDS = 12.0
_ASSIGNMENT_RETRY_DELAY = 20.0
_ASSIGNMENT_REFRESH_RETRY_DELAY = 5.0
_BUILD_REFRESH_RETRY_DELAY = 5.0
_FOOD_STALL_SECONDS = 30.0


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
    name: Tech(
        goto_key=spec.goto_key,
        research_key=spec.key,
        goto_modifiers=spec.goto_modifiers,
        requires=next(iter(spec.requires), ""),
        food=spec.price("food"),
        gold=spec.price("gold"),
        wood=spec.price("wood"),
    )
    for name, spec in RESEARCH.items()
}


@dataclass(frozen=True, slots=True)
class _PendingResearch:
    """A research awaiting confirmation from the HUD resource drop."""

    name: str
    tech: Tech
    before: Mapping[str, int]
    # Monotonic instant after which an undecided reading is judged anyway.
    settle_deadline: float = 0.0
    operation_id: int = 0
    noted_at_snapshot: int = 0
    spend_revision: int = 0


@dataclass(frozen=True, slots=True)
class _PendingTraining:
    unit: str
    cost: tuple[tuple[str, int], ...]
    before: Mapping[str, int]
    settle_deadline: float
    operation_id: int
    noted_at_snapshot: int = 0
    spend_revision: int = 0


@dataclass(frozen=True, slots=True)
class _QueuedTraining:
    unit: str
    operation_id: int


@dataclass(frozen=True, slots=True)
class _PendingAssignment:
    operation_id: int
    resource: ResourceKind
    target_id: str
    idle_count_before: int | None
    workers_before: int | None
    noted_at_snapshot: int
    command_revision: int
    settle_deadline: float


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    operation_id: int
    action: str
    status: Literal["pending", "purchased", "confirmed", "failed", "uncertain", "cancelled"]
    detail: str


# Circuit breaker (T-530): consecutive missing settlements for one building
# class before its builds are suppressed, and how long the suppression lasts.
# Run 9: 32 identical farm attempts each burned resources.
_MISSING_STREAK_LIMIT = 3
_MISSING_SUPPRESS_SECONDS = 50.0
# Villager-order ledger (T-531). Orders lead the HUD population by the TC
# queue depth. Starting villagers are unknown until the villager HUD is read;
# the initial population can include a scout and is not a villager count.
_STARTING_VILLAGERS = 0
_VILLAGER_FOOD_COST = 50
# How `_select_villager_step` picked the villager that builds.
SelectionMode = Literal["click", "idle_press", "unknown"]

# What one HUD snapshot says about a pending placement. "undecided" is not a
# miss: the reading cannot answer yet, so the placement waits for its deadline.
Verdict = Literal["confirmed", "missing", "undecided"]


# A placement whose foundation wasn't visually confirmed, awaiting settlement
# against the HUD wood spend (2026-07-11 run 2, F-11: YOLO can't see foundations,
# so a fresh rescan reports almost every REAL placement as failed — that false
# negative caused a duplicate mill).
@dataclass(frozen=True, slots=True)
class _PendingPlacement:
    building_class: str
    wood_cost: int
    wood_before: int  # wood per the HUD snapshot when the placement was made
    noted_at_snapshot: int  # snapshot_count the wood_before reading belongs to
    operation_id: int = 0
    preexisting_entity_ids: frozenset[str] = frozenset()
    spend_revision: int = 0
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


@dataclass(frozen=True, slots=True)
class _PurchasedBuilding:
    operation_id: int
    preexisting_entity_ids: frozenset[str]
    cap_before: int = 0


@dataclass
class ActionLedger:
    """Per-game observations, commitments, and settled action outcomes.

    Confirmed buildings require purchase or visual-placement evidence. Detector
    sightings alone never unlock a prerequisite. Unknown HUD fields remain
    unknown, so a purchase without a known baseline is not attempted.
    """

    population: tuple[int, int] | None = None
    last_known_population: int | None = None
    villagers: int | None = None
    last_known_villagers: int | None = None
    worker_counts: dict[str, int] = field(default_factory=dict)
    resources: dict[str, int] | None = None
    # This iteration's idle-villager reading, and how the last build acted on
    # it — see _select_villager_step, which owns both.
    idle_present: bool | None = None
    idle_count: int | None = None
    pending_assignment: _PendingAssignment | None = None
    assignment_suppressed_until: dict[ResourceKind, float] = field(default_factory=dict)
    food_progress_at: float | None = None
    selected_by: SelectionMode = "unknown"
    selected_unit: str | None = None
    selected_at_revision: int = -1
    buildings_confirmed: set[str] = field(default_factory=set)
    # A wood drop proves placement was purchased, not that construction is
    # complete. These classes still block duplicates until a new matching
    # building entity is observed after the purchase.
    building_purchases: dict[str, _PurchasedBuilding] = field(default_factory=dict)
    # Frames each gate-relevant building class has been detected in —
    # informational only (the context line reports classes past
    # _SIGHTING_MIN_FRAMES as unverified sightings).
    building_sightings: dict[str, int] = field(default_factory=dict)
    pending_placements: list[_PendingPlacement] = field(default_factory=list)
    # Circuit breaker (T-530): consecutive missing settlements per class, and
    # the monotonic instant until which a repeatedly-missing class stays blocked.
    missing_streaks: dict[str, int] = field(default_factory=dict)
    suppressed_until: dict[str, float] = field(default_factory=dict)
    # The research counterparts: awaiting settlement, proven paid for, and the
    # instant until which a proven miss stays blocked.
    pending_research: list[_PendingResearch] = field(default_factory=list)
    # A paid age-up is underway, not a completed age. Keep its action ID until
    # the age reading changes, without continuing to reserve already-spent food.
    age_up_paid: dict[str, int] = field(default_factory=dict)
    # Non-age technology has no completion telemetry. Payment suppresses a
    # duplicate press, but never fabricates a completed-research observation.
    research_purchases: set[str] = field(default_factory=set)
    pending_training: list[_PendingTraining] = field(default_factory=list)
    queued_training: list[_QueuedTraining] = field(default_factory=list)
    researched: set[str] = field(default_factory=set)
    research_blocked_until: dict[str, float] = field(default_factory=dict)
    snapshot_count: int = 0
    # Villagers ordered so far — a strategy signal, not the capacity guard.
    villagers_ordered: int = _STARTING_VILLAGERS
    # Validated age from GameState (strategist OCR), synced once per turn —
    # selects the villager order target and the rejection message (T-538).
    current_age: str = "Dark Age"
    age_known: bool = False
    input_revision: int = 0
    hud_revision: int = -1
    failure_streak: int = 0
    failure_count: int = 0
    recent_failures: list[str] = field(default_factory=list)
    retry_after: dict[str, float] = field(default_factory=dict)
    outcomes: list[ActionOutcome] = field(default_factory=list)
    next_operation_id: int = 1
    known_resources: frozenset[str] = frozenset()
    population_known: bool = False
    spatial_valid: bool = True
    claimed_spend: dict[tuple[str, int, int], int] = field(default_factory=dict)
    uncertain_operations: set[int] = field(default_factory=set)
    outcome_screenshots: dict[int, str] = field(default_factory=dict)
    attempted_economic_ids: set[int] = field(default_factory=set)
    confirmed_economic_ids: set[int] = field(default_factory=set)
    food_gathered: int = 0
    tc_progress_at: float | None = None

    def new_operation_id(self) -> int:
        operation_id = self.next_operation_id
        self.next_operation_id += 1
        return operation_id

    def has_pending_since(self, first_operation_id: int) -> bool:
        """Whether this action left a new operation awaiting observation."""
        return (
            self.pending_assignment is not None
            and self.pending_assignment.operation_id >= first_operation_id
        ) or any(
            item.operation_id >= first_operation_id
            for items in (self.pending_placements, self.pending_research, self.pending_training)
            for item in items
        )

    def reservations(self) -> dict[str, int]:
        """Commitments not yet reconciled against a fresh HUD reading."""
        reserved: dict[str, int] = {"wood": sum(p.wood_cost for p in self.pending_placements)}
        for pending in self.pending_research:
            for resource, amount in _research_costs(pending.tech).items():
                reserved[resource] = reserved.get(resource, 0) + amount
        for pending in self.pending_training:
            for resource, amount in pending.cost:
                reserved[resource] = reserved.get(resource, 0) + amount
        return reserved

    def record_outcome(self, outcome: ActionOutcome) -> None:
        self.outcomes.append(outcome)
        del self.outcomes[:-20]
        if outcome.status in {"pending", "failed", "uncertain", "cancelled"}:
            self.attempted_economic_ids.add(outcome.operation_id)
        if outcome.status == "confirmed":
            self.confirmed_economic_ids.add(outcome.operation_id)
            self.uncertain_operations.discard(outcome.operation_id)
        elif outcome.status == "uncertain":
            self.uncertain_operations.add(outcome.operation_id)
            self.recent_failures.append(f"{outcome.action}: {outcome.detail}")
            del self.recent_failures[:-5]
        elif outcome.status in {"failed", "cancelled"}:
            self.uncertain_operations.discard(outcome.operation_id)
        if outcome.status == "failed":
            self.record_failure(f"{outcome.action}: {outcome.detail}")
        elif outcome.status in {"purchased", "confirmed"}:
            self.record_success()
        log.info(
            "action_outcome",
            action_id=outcome.operation_id,
            action=outcome.action,
            status=outcome.status,
            detail=outcome.detail,
            reservations=self.reservations(),
        )

    def finalize_unverified(self) -> None:
        """Retain unresolved commitments and mark the recorded score invalid."""
        unresolved: list[tuple[int, str]] = [
            (pending.operation_id, f"build_{pending.building_class}")
            for pending in self.pending_placements
        ]
        unresolved.extend(
            (pending.operation_id, _research_action_id(pending.name))
            for pending in self.pending_research
        )
        unresolved.extend(
            (
                pending.operation_id,
                "queue_villager" if pending.unit == "villager" else f"train_{pending.unit}",
            )
            for pending in self.pending_training
        )
        unresolved.extend(
            (
                queued.operation_id,
                "queue_villager" if queued.unit == "villager" else f"train_{queued.unit}",
            )
            for queued in self.queued_training
        )
        unresolved.extend(
            (purchase.operation_id, f"build_{name}")
            for name, purchase in self.building_purchases.items()
        )
        unresolved.extend(
            (operation_id, _research_action_id(name))
            for name, operation_id in self.age_up_paid.items()
        )
        if self.pending_assignment is not None:
            unresolved.append(
                (
                    self.pending_assignment.operation_id,
                    f"assign_{self.pending_assignment.resource}",
                )
            )
        for operation_id, action in unresolved:
            if operation_id not in self.uncertain_operations:
                self.record_outcome(
                    ActionOutcome(
                        operation_id,
                        action,
                        "uncertain",
                        "run ended before the intended effect could be verified",
                    )
                )

    def record_failure(self, reason: str) -> None:
        self.failure_streak += 1
        self.failure_count += 1
        self.recent_failures.append(reason)
        del self.recent_failures[:-5]

    def record_success(self) -> None:
        self.failure_streak = 0

    def defer_action(self, action: str, seconds: float, reason: str) -> None:
        """Avoid repeating an action before its missing evidence can change."""
        self.retry_after[action] = _now() + seconds
        log.info("action_retry_deferred", action=action, seconds=seconds, reason=reason)

    def note_input(self) -> None:
        self.input_revision += 1

    def record_interrupted_purchase(
        self,
        operation_id: int,
        action: str,
        before_revision: int,
        detail: str,
        *,
        before_spend_status: Literal["cancelled", "failed"] = "cancelled",
    ) -> None:
        """Release an unspent commitment; retain a possibly spent one."""
        pending = any(
            item.operation_id == operation_id
            for items in (self.pending_placements, self.pending_research, self.pending_training)
            for item in items
        )
        if not pending:
            return  # a concurrent observation already settled it
        if self.input_revision == before_revision:
            self.pending_placements = [
                item for item in self.pending_placements if item.operation_id != operation_id
            ]
            self.pending_research = [
                item for item in self.pending_research if item.operation_id != operation_id
            ]
            self.pending_training = [
                item for item in self.pending_training if item.operation_id != operation_id
            ]
            status: Literal["cancelled", "failed", "uncertain"] = before_spend_status
        else:
            status = "uncertain"
        self.record_outcome(ActionOutcome(operation_id, action, status, detail))


_build_gates = ActionLedger()
_active_ledger: ContextVar[ActionLedger | None] = ContextVar("active_action_ledger", default=None)


def current_ledger() -> ActionLedger:
    """Return the ledger bound to this game and inherited by its async tasks."""
    return _active_ledger.get() or _build_gates


def bind_ledger(ledger: ActionLedger) -> Token[ActionLedger | None]:
    return _active_ledger.set(ledger)


def unbind_ledger(token: Token[ActionLedger | None]) -> None:
    _active_ledger.reset(token)


def ledger_policy_state(ledger: ActionLedger | None = None) -> PolicyState:
    """The latest observed facts plus unsettled commitments for input preflight."""
    ledger = ledger or current_ledger()
    population, cap = ledger.population or (0, 0)
    now = _now()
    return PolicyState(
        age=ledger.current_age,
        age_known=ledger.age_known,
        food=(ledger.resources or {}).get("food", 0),
        wood=(ledger.resources or {}).get("wood", 0),
        gold=(ledger.resources or {}).get("gold", 0),
        stone=(ledger.resources or {}).get("stone", 0),
        population=population,
        population_cap=cap,
        population_known=ledger.population_known,
        villagers=ledger.villagers,
        villagers_ordered=ledger.villagers_ordered,
        villager_jobs=ledger.worker_counts,
        buildings_seen=frozenset(ledger.buildings_confirmed),
        pending_buildings=frozenset(p.building_class for p in ledger.pending_placements)
        | frozenset(ledger.building_purchases),
        building_purchases=frozenset(ledger.building_purchases),
        pending_research=frozenset(p.name for p in ledger.pending_research)
        | frozenset(ledger.age_up_paid),
        age_up_paid=frozenset(ledger.age_up_paid),
        research_purchases=frozenset(ledger.research_purchases),
        researched=frozenset(ledger.researched),
        pending_actions=frozenset(
            [f"build_{p.building_class}" for p in ledger.pending_placements]
            + [f"build_{name}" for name in ledger.building_purchases]
            + [_research_action_id(p.name) for p in ledger.pending_research]
            + [_research_action_id(name) for name in ledger.age_up_paid]
            + [_research_action_id(name) for name in ledger.research_purchases]
            + [
                "queue_villager" if p.unit == "villager" else f"train_{p.unit}"
                for p in ledger.pending_training
            ]
        ),
        suppressed_actions=frozenset(
            [f"build_{name}" for name, until in ledger.suppressed_until.items() if until > now]
            + [
                f"assign_{kind}"
                for kind, until in ledger.assignment_suppressed_until.items()
                if until > now
            ]
            + [
                _research_action_id(name)
                for name, until in ledger.research_blocked_until.items()
                if until > now
            ]
            + [action for action, until in ledger.retry_after.items() if until > now]
        ),
        reserved_resources=ledger.reservations(),
        pending_population=len(ledger.pending_training) + len(ledger.queued_training),
        pending_villagers=sum(p.unit == "villager" for p in ledger.pending_training)
        + sum(p.unit == "villager" for p in ledger.queued_training),
        assignment_pending=ledger.pending_assignment is not None,
        food_stalled=(
            "food" in ledger.known_resources
            and ledger.food_progress_at is not None
            and now - ledger.food_progress_at >= _FOOD_STALL_SECONDS
        ),
        tc_stalled=(
            ledger.villagers is not None
            and ledger.tc_progress_at is not None
            and now - ledger.tc_progress_at >= _FOOD_STALL_SECONDS
        ),
        known_resources=ledger.known_resources,
        idle_present=ledger.idle_present,
        idle_count=ledger.idle_count,
        visible_classes=frozenset(str(entity.get("class", "")) for entity in _detected_entities),
        spatial_valid=ledger.spatial_valid,
    )


def observe_hud(
    population: int,
    population_cap: int,
    resources: Mapping[str, int],
    *,
    idle_present: bool | None = None,
    idle_count: int | None = None,
    known_resources: frozenset[str] | None = None,
    population_known: bool = True,
    input_revision: int | None = None,
    villagers: int | None = None,
    worker_counts: Mapping[str, int] | None = None,
) -> None:
    """Feed this turn's HUD reading into the build gates.

    Settle pending operations against the previous HUD baseline before replacing
    that baseline. Unknown fields remain unknown; a small net drop is not proof
    that a purchase failed or succeeded.
    """
    current_ledger().snapshot_count += 1
    fresh_resources = (
        resources
        if known_resources is None
        else {name: value for name, value in resources.items() if name in known_resources}
    )
    ledger = current_ledger()
    food_now = fresh_resources.get("food")
    food_before = (ledger.resources or {}).get("food")
    if food_now is not None and food_before is None:
        ledger.food_progress_at = _now()
    if villagers is not None and ledger.last_known_villagers is None:
        ledger.tc_progress_at = _now()
    # Both readings are passed in, not read off the gates: settlement runs
    # BEFORE the snapshot is replaced, which is the ordering the deltas need.
    claimed_spend = current_ledger().claimed_spend
    food_claimed_before = sum(
        amount for (kind, _, _), amount in claimed_spend.items() if kind == "food"
    )
    _settle_pending_placements(
        fresh_resources.get("wood"),
        population_cap if population_known else None,
        claimed_spend,
        input_revision=input_revision,
    )
    purchased_house = ledger.building_purchases.get(_HOUSE_CLASS)
    if (
        purchased_house is not None
        and population_known
        and population_cap >= purchased_house.cap_before + _HOUSE_CAP_STEP
    ):
        record_confirmed_buildings([_HOUSE_CLASS])
    _settle_pending_research(fresh_resources, claimed_spend, input_revision=input_revision)
    _settle_pending_training(
        fresh_resources,
        claimed_spend,
        input_revision=input_revision,
        villagers_now=villagers,
        population_now=population if population_known else None,
    )
    _reconcile_queued_training(population if population_known else None, villagers)
    _settle_pending_assignment(
        idle_present,
        idle_count,
        worker_counts,
        input_revision=input_revision,
    )
    food_claimed_after = sum(
        amount for (kind, _, _), amount in claimed_spend.items() if kind == "food"
    )
    ambiguous_food = any(
        any(kind == "food" for kind, _ in pending.cost) for pending in ledger.pending_training
    ) or any(pending.tech.food for pending in ledger.pending_research)
    if food_now is not None and food_before is not None and not ambiguous_food:
        income = food_now - food_before + food_claimed_after - food_claimed_before
        if 0 < income <= 300:
            ledger.food_gathered += income
            ledger.food_progress_at = _now()
    active_snapshots = (
        [p.noted_at_snapshot for p in current_ledger().pending_placements]
        + [p.noted_at_snapshot for p in current_ledger().pending_research]
        + [p.noted_at_snapshot for p in current_ledger().pending_training]
    )
    oldest = min(active_snapshots, default=current_ledger().snapshot_count + 1)
    current_ledger().claimed_spend = {
        key: amount for key, amount in claimed_spend.items() if key[2] >= oldest
    }
    current_ledger().population = (population, population_cap)
    current_ledger().population_known = population_known
    if population_known:
        current_ledger().last_known_population = population
    current_ledger().resources = dict(resources)
    current_ledger().known_resources = (
        frozenset(resources) if known_resources is None else known_resources
    )
    current_ledger().idle_present = idle_present
    current_ledger().idle_count = idle_count
    if villagers is not None:
        if ledger.last_known_villagers is not None and villagers > ledger.last_known_villagers:
            ledger.tc_progress_at = _now()
        ledger.last_known_villagers = villagers
    ledger.villagers = villagers
    ledger.worker_counts = dict(worker_counts or {})
    ledger.hud_revision = ledger.input_revision if input_revision is None else input_revision
    if villagers is not None:
        ledger.villagers_ordered = (
            villagers
            + sum(pending.unit == "villager" for pending in ledger.pending_training)
            + sum(queued.unit == "villager" for queued in ledger.queued_training)
        )


def _settle_pending_assignment(
    idle_present: bool | None,
    idle_count: int | None,
    worker_counts: Mapping[str, int] | None = None,
    *,
    input_revision: int | None,
) -> None:
    """Require a matching worker-count increase, using idle evidence when readable."""
    ledger = current_ledger()
    pending = ledger.pending_assignment
    if pending is None or ledger.snapshot_count <= pending.noted_at_snapshot:
        return
    observed_revision = ledger.input_revision if input_revision is None else input_revision
    if observed_revision < pending.command_revision:
        return
    workers_now = None if worker_counts is None else worker_counts.get(pending.resource)
    idle_decreased = idle_present is False or (
        idle_count is not None
        and pending.idle_count_before is not None
        and idle_count < pending.idle_count_before
    )
    idle_unreadable = idle_present is None and idle_count is None
    if (
        workers_now is not None
        and pending.workers_before is not None
        and workers_now > pending.workers_before
        and (idle_decreased or idle_unreadable)
    ):
        ledger.pending_assignment = None
        ledger.record_outcome(
            ActionOutcome(
                pending.operation_id,
                f"assign_{pending.resource}",
                "confirmed",
                "worker count increased with no contradictory idle evidence",
            )
        )
        return
    if _now() < pending.settle_deadline:
        return
    # An inconclusive assignment must not monopolize the single dispatch slot.
    # Keep its outcome uncertain for scoring, then let a later verified idle
    # worker be retried after the resource-specific cooldown.
    ledger.pending_assignment = None
    ledger.assignment_suppressed_until[pending.resource] = _now() + _ASSIGNMENT_RETRY_DELAY
    if pending.operation_id in ledger.uncertain_operations:
        return
    ledger.record_outcome(
        ActionOutcome(
            pending.operation_id,
            f"assign_{pending.resource}",
            "uncertain",
            f"worker assignment to {pending.target_id} not confirmed",
        )
    )


def observe_age(age: str | None, *, input_revision: int | None = None) -> None:
    """Sync the observed age; a completed advancement supersedes HUD settlement."""
    current_ledger().age_known = bool(age)
    if age:
        ledger = current_ledger()
        ledger.current_age = age
        result_ages = {
            "feudal_age": "Feudal Age",
            "castle_age": "Castle Age",
            "imperial_age": "Imperial Age",
        }
        completed = [
            pending
            for pending in ledger.pending_research
            if result_ages.get(pending.name) == age
            and (input_revision is None or input_revision >= pending.spend_revision)
        ]
        if completed:
            ledger.pending_research = [
                pending for pending in ledger.pending_research if pending not in completed
            ]
            for pending in completed:
                ledger.researched.add(pending.name)
                ledger.record_outcome(
                    ActionOutcome(
                        pending.operation_id,
                        _research_action_id(pending.name),
                        "confirmed",
                        "age advancement observed",
                    )
                )
        for name, target_age in result_ages.items():
            operation_id = ledger.age_up_paid.pop(name, None) if target_age == age else None
            if operation_id is not None:
                ledger.researched.add(name)
                ledger.record_outcome(
                    ActionOutcome(
                        operation_id,
                        _research_action_id(name),
                        "confirmed",
                        "age advancement observed",
                    )
                )


def _note_pending_placement(
    building_key: str,
    *,
    menu: str = ECON_MENU,
    point: tuple[int, int] = (0, 0),
    wood_before: int | None = None,
    cap_before: int | None = None,
    noted_at_snapshot: int | None = None,
) -> _PendingPlacement | None:
    """Queue an unconfirmed placement for wood-delta settlement next snapshot."""
    cls = building_class(menu, building_key)
    cost = _WOOD_COST_BY_CLASS.get(cls or "")
    if wood_before is None:
        wood_before = (current_ledger().resources or {}).get("wood")
    if (
        cls is None
        or cost is None
        or wood_before is None
        or current_ledger().hud_revision != current_ledger().input_revision
        or "wood" not in current_ledger().known_resources
    ):
        # No wood baseline to settle against — the placement stays unconfirmed
        # for good, despite the caller's "settled next turn" detail. Say so.
        log.debug("placement_pending_dropped", building_key=building_key)
        return None
    if cap_before is None:
        _, cap_before = current_ledger().population or (0, 0)
    is_house = cls == _HOUSE_CLASS
    placement = _PendingPlacement(
        building_class=cls,
        wood_cost=cost,
        wood_before=wood_before,
        noted_at_snapshot=(
            current_ledger().snapshot_count if noted_at_snapshot is None else noted_at_snapshot
        ),
        cap_before=cap_before,
        selected_by=current_ledger().selected_by,
        point=point,
        settle_deadline=_now() + (_HOUSE_SETTLE_SECONDS if is_house else _PLACEMENT_SETTLE_SECONDS),
        operation_id=current_ledger().new_operation_id(),
        preexisting_entity_ids=frozenset(
            entity_id
            for entity in _detected_entities
            if entity.get("class") == cls and isinstance(entity_id := entity.get("id"), str)
        ),
        spend_revision=current_ledger().input_revision + 1,
    )
    current_ledger().pending_placements.append(placement)
    current_ledger().record_outcome(
        ActionOutcome(placement.operation_id, f"build_{cls}", "pending", "awaiting HUD settlement")
    )
    return placement


def _settle_pending_placements(
    wood_now: int | None,
    cap_now: int | None,
    claimed_spend: dict[tuple[str, int, int], int] | None = None,
    *,
    input_revision: int | None = None,
) -> None:
    """Settle purchases from full HUD cost or completed-house cap evidence.

    An inconclusive reading stays pending after its deadline and records an
    uncertain outcome. The reservation and duplicate guard remain until
    independent evidence resolves the operation.
    """
    if not current_ledger().pending_placements:
        return
    still_pending: list[_PendingPlacement] = []
    outcomes: list[ActionOutcome] = []
    spent = claimed_spend if claimed_spend is not None else {}
    claimed_cap: dict[int, int] = {}
    for pending in current_ledger().pending_placements:
        if input_revision is not None and input_revision < pending.spend_revision:
            still_pending.append(pending)
            continue
        completed_house = (
            pending.is_house and _house_verdict(pending, cap_now, claimed_cap) == "confirmed"
        )
        verdict = (
            "confirmed"
            if completed_house
            else _wood_verdict(pending, wood_now, spent)
            if wood_now is not None
            else "undecided"
        )
        if verdict == "undecided":
            still_pending.append(pending)
            if (
                _now() >= pending.settle_deadline
                and pending.operation_id not in current_ledger().uncertain_operations
            ):
                outcomes.append(
                    ActionOutcome(
                        pending.operation_id,
                        f"build_{pending.building_class}",
                        "uncertain",
                        "purchase not verifiable from HUD or building evidence",
                    )
                )
            continue
        evidence = _settlement_evidence(pending, wood_now, cap_now)
        if verdict == "confirmed":
            if completed_house:
                # Population capacity only rises after a house is complete.
                record_confirmed_buildings([pending.building_class])
            else:
                spec = BY_ID[f"build_{pending.building_class}"]
                if spec.unique or pending.is_house:
                    # Unique prerequisites stay unavailable until construction
                    # is seen. Repeatable farms only need purchase settlement;
                    # another farm can be started while the first is building.
                    current_ledger().building_purchases[pending.building_class] = (
                        _PurchasedBuilding(
                            pending.operation_id, pending.preexisting_entity_ids, pending.cap_before
                        )
                    )
                _clear_missing_streak(pending.building_class)
            log.info("build_purchase_confirmed", **evidence)
            outcomes.append(
                ActionOutcome(
                    pending.operation_id,
                    f"build_{pending.building_class}",
                    "confirmed" if completed_house else "purchased",
                    "house capacity observed" if completed_house else "HUD purchase confirmed",
                )
            )
        else:
            _note_missing_settlement(pending.building_class)
            outcomes.append(
                ActionOutcome(
                    pending.operation_id,
                    f"build_{pending.building_class}",
                    "failed",
                    "HUD purchase missing",
                )
            )
            log.warning(
                "build_purchase_missing",
                **evidence,
                selected_by=pending.selected_by,
                x=pending.point[0],
                y=pending.point[1],
            )
    current_ledger().pending_placements = still_pending
    for outcome in outcomes:
        current_ledger().record_outcome(outcome)


def _house_verdict(
    pending: _PendingPlacement, cap_now: int | None, claimed: dict[int, int]
) -> Verdict:
    """Confirmed once the population cap has risen by a whole house.

    Never "missing": an unmoved cap means the house may still be under
    construction. `claimed` stops one +10 jump from confirming three pending
    houses — the cap analogue of `spend_by_baseline` (run 3, F-17).
    """
    if cap_now is None:
        return "undecided"
    already = claimed.get(pending.cap_before, 0)
    if cap_now - pending.cap_before - already < _HOUSE_CAP_STEP:
        return "undecided"
    claimed[pending.cap_before] = already + _HOUSE_CAP_STEP
    return "confirmed"


def _wood_verdict(
    pending: _PendingPlacement, wood_now: int, spend_by_baseline: dict[tuple[str, int, int], int]
) -> Verdict:
    """Whether the HUD wood delta covers this placement's cost.

    An unchanged reading is stale OCR, not a miss.

    Only a full net cost proves payment. Gathering can hide some or all of a
    purchase, so a smaller drop remains undecided. Claimed spend is deducted
    per shared baseline so one drop cannot confirm multiple operations.
    """
    baseline = ("wood", pending.wood_before, pending.noted_at_snapshot)
    spent = spend_by_baseline.get(baseline, 0)
    if pending.wood_before - wood_now - spent < pending.wood_cost:
        return "undecided"
    spend_by_baseline[baseline] = spent + pending.wood_cost
    return "confirmed"


def _settlement_evidence(
    pending: _PendingPlacement, wood_now: int | None, cap_now: int | None
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
    }


def _note_pending_research(name: str, tech: Tech) -> _PendingResearch | None:
    """Queue a research for HUD settlement next snapshot."""
    before = current_ledger().resources
    if (
        before is None
        or current_ledger().hud_revision != current_ledger().input_revision
        or any(kind not in current_ledger().known_resources for kind in _research_costs(tech))
    ):
        log.debug("research_pending_dropped", tech=name)
        return None
    pending = _PendingResearch(
        name=name,
        tech=tech,
        before=MappingProxyType(dict(before)),
        settle_deadline=_now() + _RESEARCH_SETTLE_SECONDS,
        operation_id=current_ledger().new_operation_id(),
        noted_at_snapshot=current_ledger().snapshot_count,
        spend_revision=current_ledger().input_revision + 1,
    )
    current_ledger().pending_research.append(pending)
    current_ledger().record_outcome(
        ActionOutcome(
            pending.operation_id,
            _research_action_id(name),
            "pending",
            "awaiting HUD settlement",
        )
    )
    return pending


def _settle_pending_research(
    resources: Mapping[str, int],
    claimed_spend: dict[tuple[str, int, int], int] | None = None,
    *,
    input_revision: int | None = None,
) -> None:
    """Confirm or report each pending research against the HUD resource drop.

    This is the feedback a raw `press` never had: a keystroke always "succeeds",
    so a greyed-out button looked identical to a working one. Run 2026_08_21_2
    pressed the age-up key 10 times over 4 minutes on that blind spot.
    """
    if not current_ledger().pending_research:
        return
    still_pending: list[_PendingResearch] = []
    outcomes: list[ActionOutcome] = []
    claimed = claimed_spend if claimed_spend is not None else {}
    for pending in current_ledger().pending_research:
        if input_revision is not None and input_revision < pending.spend_revision:
            still_pending.append(pending)
            continue
        verdict = _research_verdict(pending, resources, claimed)
        if verdict == "undecided":
            still_pending.append(pending)
            if (
                _now() >= pending.settle_deadline
                and pending.operation_id not in current_ledger().uncertain_operations
            ):
                outcomes.append(
                    ActionOutcome(
                        pending.operation_id,
                        _research_action_id(pending.name),
                        "uncertain",
                        "research purchase not verifiable from HUD",
                    )
                )
            continue
        if verdict == "confirmed":
            age_up = pending.name in {"feudal_age", "castle_age", "imperial_age"}
            if age_up:
                current_ledger().age_up_paid[pending.name] = pending.operation_id
            else:
                current_ledger().research_purchases.add(pending.name)
            log.info("research_purchase_confirmed", tech=pending.name, age_up=age_up)
            outcomes.append(
                ActionOutcome(
                    pending.operation_id,
                    _research_action_id(pending.name),
                    "purchased",
                    "HUD purchase confirmed; awaiting observed age"
                    if age_up
                    else "HUD purchase confirmed; completion unobserved",
                )
            )
        else:
            current_ledger().research_blocked_until[pending.name] = (
                _now() + _MISSING_SUPPRESS_SECONDS
            )
            outcomes.append(
                ActionOutcome(
                    pending.operation_id,
                    _research_action_id(pending.name),
                    "failed",
                    "HUD purchase missing",
                )
            )
            log.warning(
                "research_missing",
                tech=pending.name,
                retry_in_s=round(_MISSING_SUPPRESS_SECONDS),
                **_research_costs(pending.tech),
            )
    current_ledger().pending_research = still_pending
    for outcome in outcomes:
        current_ledger().record_outcome(outcome)


def _note_pending_training(unit: str) -> _PendingTraining | None:
    before = current_ledger().resources
    spec = UNITS[unit]
    if (
        before is None
        or current_ledger().hud_revision != current_ledger().input_revision
        or any(kind not in current_ledger().known_resources for kind, _ in spec.cost)
    ):
        return None
    pending = _PendingTraining(
        unit=unit,
        cost=spec.cost,
        before=MappingProxyType(dict(before)),
        settle_deadline=_now() + _RESEARCH_SETTLE_SECONDS,
        operation_id=current_ledger().new_operation_id(),
        noted_at_snapshot=current_ledger().snapshot_count,
        spend_revision=current_ledger().input_revision + 1,
    )
    current_ledger().pending_training.append(pending)
    current_ledger().record_outcome(
        ActionOutcome(pending.operation_id, f"train_{unit}", "pending", "awaiting HUD settlement")
    )
    return pending


def _settle_pending_training(
    resources: Mapping[str, int],
    claimed_spend: dict[tuple[str, int, int], int] | None = None,
    *,
    input_revision: int | None = None,
    villagers_now: int | None = None,
    population_now: int | None = None,
) -> None:
    still_pending: list[_PendingTraining] = []
    outcomes: list[ActionOutcome] = []
    claimed = claimed_spend if claimed_spend is not None else {}
    ledger = current_ledger()
    villager_gain = (
        max(villagers_now - ledger.last_known_villagers, 0)
        if villagers_now is not None and ledger.last_known_villagers is not None
        else 0
    )
    population_gain = (
        max(population_now - ledger.last_known_population, 0)
        if population_now is not None and ledger.last_known_population is not None
        else 0
    )
    available_villagers = max(
        villager_gain - sum(queued.unit == "villager" for queued in ledger.queued_training), 0
    )
    available_military = max(
        population_gain
        - villager_gain
        - sum(queued.unit != "villager" for queued in ledger.queued_training),
        0,
    )
    for pending in current_ledger().pending_training:
        if input_revision is not None and input_revision < pending.spend_revision:
            still_pending.append(pending)
            continue
        verdict = _cost_verdict(
            pending.cost, pending.before, resources, pending.noted_at_snapshot, claimed
        )
        delivery_proves_purchase = (pending.unit == "villager" and available_villagers > 0) or (
            pending.unit != "villager" and available_military > 0
        )
        if delivery_proves_purchase:
            if pending.unit == "villager":
                available_villagers -= 1
            else:
                available_military -= 1
            if verdict != "confirmed":
                for kind, price in pending.cost:
                    key = (kind, pending.before[kind], pending.noted_at_snapshot)
                    claimed[key] = claimed.get(key, 0) + price
            verdict = "confirmed"
        if verdict == "undecided":
            still_pending.append(pending)
            if (
                _now() >= pending.settle_deadline
                and pending.operation_id not in current_ledger().uncertain_operations
            ):
                outcomes.append(
                    ActionOutcome(
                        pending.operation_id,
                        "queue_villager" if pending.unit == "villager" else f"train_{pending.unit}",
                        "uncertain",
                        "training purchase not verifiable from HUD",
                    )
                )
            continue
        status: Literal["confirmed", "failed"] = "confirmed" if verdict == "confirmed" else "failed"
        if status == "confirmed" and pending.unit == "villager":
            current_ledger().villagers_ordered += 1
        if status == "confirmed":
            current_ledger().queued_training.append(
                _QueuedTraining(pending.unit, pending.operation_id)
            )
        outcomes.append(
            ActionOutcome(
                pending.operation_id,
                "queue_villager" if pending.unit == "villager" else f"train_{pending.unit}",
                "purchased" if status == "confirmed" else "failed",
                "HUD purchase settled",
            )
        )
    current_ledger().pending_training = still_pending
    for outcome in outcomes:
        current_ledger().record_outcome(outcome)


def _reconcile_queued_training(
    population_now: int | None, villagers_now: int | None = None
) -> None:
    """Attribute villager deliveries to villagers, military to other population."""
    ledger = current_ledger()
    previous = ledger.last_known_population
    prior_villagers = ledger.last_known_villagers
    villager_gain = (
        max(villagers_now - prior_villagers, 0)
        if villagers_now is not None and prior_villagers is not None
        else 0
    )
    population_gain = (
        max(population_now - previous, 0)
        if population_now is not None and previous is not None
        else 0
    )
    military_gain = max(population_gain - villager_gain, 0)
    delivered: list[_QueuedTraining] = []
    remaining: list[_QueuedTraining] = []
    for item in ledger.queued_training:
        if item.unit == "villager" and villager_gain:
            delivered.append(item)
            villager_gain -= 1
        elif item.unit != "villager" and military_gain:
            delivered.append(item)
            military_gain -= 1
        else:
            remaining.append(item)
    ledger.queued_training = remaining
    for item in delivered:
        ledger.record_outcome(
            ActionOutcome(
                item.operation_id,
                "queue_villager" if item.unit == "villager" else f"train_{item.unit}",
                "confirmed",
                "villager HUD count increased"
                if item.unit == "villager"
                else "military population delivered",
            )
        )


def _cost_verdict(
    cost: tuple[tuple[str, int], ...],
    before: Mapping[str, int],
    resources: Mapping[str, int],
    snapshot_count: int,
    claimed_spend: dict[tuple[str, int, int], int] | None = None,
) -> Verdict:
    shortfalls = []
    claimed = claimed_spend if claimed_spend is not None else {}
    for kind, price in cost:
        now = resources.get(kind)
        was = before.get(kind)
        if now is None or was is None:
            return "undecided"
        shortfalls.append(was - now - claimed.get((kind, was, snapshot_count), 0) < price)
    if any(shortfalls):
        return "undecided"
    for kind, price in cost:
        baseline = before[kind]
        key = (kind, baseline, snapshot_count)
        claimed[key] = claimed.get(key, 0) + price
    return "confirmed"


def _research_verdict(
    pending: _PendingResearch,
    resources: Mapping[str, int],
    claimed_spend: dict[tuple[str, int, int], int] | None = None,
) -> Verdict:
    """Confirmed once every cost resource has fallen far enough to have paid.

    ANY unchanged cost reading is stale OCR, not a refusal: a frame where food
    updated but gold did not once reported a paid-for age-up as missing.
    """
    return _cost_verdict(
        tuple(_research_costs(pending.tech).items()),
        pending.before,
        resources,
        pending.noted_at_snapshot,
        claimed_spend,
    )


def _research_costs(tech: Tech) -> dict[str, int]:
    """The non-zero prices of one technology, keyed by resource."""
    return {
        kind: price
        for kind, price in (("food", tech.food), ("gold", tech.gold), ("wood", tech.wood))
        if price
    }


def _research_action_id(name: str) -> str:
    return RESEARCH[name].id


def _is_pop_capped() -> bool:
    """Whether the HUD shows no population headroom. False with no reading yet."""
    population_reading = current_ledger().population
    if population_reading is None:
        return False
    population, cap = population_reading
    return cap > 0 and population >= cap


def is_pop_capped() -> bool:
    """Whether the HUD says no villager can be queued. False when unread."""
    return _is_pop_capped()


def _clear_missing_streak(building_class: str) -> None:
    """A real purchase proves the build path works — lift any suppression."""
    current_ledger().missing_streaks.pop(building_class, None)
    current_ledger().suppressed_until.pop(building_class, None)


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
    streak = current_ledger().missing_streaks.get(building_class, 0) + 1
    current_ledger().missing_streaks[building_class] = streak
    if streak >= _MISSING_STREAK_LIMIT:
        current_ledger().suppressed_until[building_class] = _now() + _MISSING_SUPPRESS_SECONDS
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
        current_ledger().building_sightings[cls] = (
            current_ledger().building_sightings.get(cls, 0) + 1
        )


def record_confirmed_buildings(classes: Iterable[str]) -> None:
    """Remember gate-relevant building classes observed as standing.

    A paid placement alone is not a completed prerequisite. The only other
    completion signal is a house's observed population-cap increase.
    Proof the build path works also lifts any T-530 suppression: run 13
    (F-45) kept a class suppressed after it was verified standing because
    only the wood-delta path cleared the streak."""
    proven = set(classes) & GATE_BUILDING_CLASSES
    current_ledger().buildings_confirmed.update(proven)
    for cls in proven:
        purchased = current_ledger().building_purchases.pop(cls, None)
        if purchased is not None:
            current_ledger().record_outcome(
                ActionOutcome(
                    purchased.operation_id,
                    f"build_{cls}",
                    "confirmed",
                    "new building entity observed",
                )
            )
        _clear_missing_streak(cls)


def record_observed_buildings(entities: Iterable[tuple[str, str]]) -> None:
    """Promote a paid placement only when a matching building ID is new.

    Ownership classification currently covers military units, not buildings.
    The caller excludes any entity explicitly identified as enemy-owned.
    """
    ledger = current_ledger()
    observed = tuple(entities)
    purchased = ledger.building_purchases
    completed = {
        cls
        for entity_id, cls in observed
        if cls != _HOUSE_CLASS
        and cls in purchased
        and entity_id not in purchased[cls].preexisting_entity_ids
    }
    record_confirmed_buildings(completed)
    directly_completed = [
        pending
        for pending in ledger.pending_placements
        if any(
            cls == pending.building_class
            and cls != _HOUSE_CLASS
            and entity_id not in pending.preexisting_entity_ids
            for entity_id, cls in observed
        )
    ]
    if directly_completed:
        ledger.pending_placements = [
            pending for pending in ledger.pending_placements if pending not in directly_completed
        ]
        for pending in directly_completed:
            record_confirmed_buildings([pending.building_class])
            ledger.record_outcome(
                ActionOutcome(
                    pending.operation_id,
                    f"build_{pending.building_class}",
                    "confirmed",
                    "new building entity observed",
                )
            )


def confirmed_buildings() -> frozenset[str]:
    """Building classes the agent provably built this game (purchase-grade
    evidence) — copied into GameState each turn so the reactive tier can gate
    Feudal prep and the age-up press on the two-building requirement."""
    return frozenset(current_ledger().buildings_confirmed)


def villagers_ordered() -> int:
    """Observed villagers plus confirmed, undelivered orders."""
    return current_ledger().villagers_ordered


def villager_queue_rejection() -> str | None:
    """Explain the hard population and food gates, never a strategy target."""
    ledger = current_ledger()
    ordered = ledger.villagers_ordered
    population, cap = ledger.population or (0, 0)
    if not ledger.population_known:
        reason = "population capacity has not been observed"
    elif population + len(ledger.pending_training) + len(ledger.queued_training) >= cap:
        reason = f"population capacity reached ({population}/{cap}; {ordered} ordered)"
    elif "food" not in ledger.known_resources:
        reason = "food has not been observed"
    else:
        food = (ledger.resources or {}).get("food")
        if food is not None and food - ledger.reservations().get("food", 0) >= _VILLAGER_FOOD_COST:
            return None
        reason = f"villager costs {_VILLAGER_FOOD_COST} food, you have {food}"
    log.info("villager_queue_rejected", reason=reason, ordered=ordered)
    return reason


def sighted_buildings() -> frozenset[str]:
    """Building classes detection has seen persistently but nothing proved —
    context-line information only, never gate evidence (F-36)."""
    return frozenset(
        cls
        for cls, frames in current_ledger().building_sightings.items()
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
        for cls, until in sorted(current_ledger().suppressed_until.items())
        if now < until
    ]
    blocked += [
        f"{name} (retryable in {round(until - now)}s)"
        for name, until in sorted(current_ledger().research_blocked_until.items())
        if now < until
    ]
    blocked += [f"{name} (already researched)" for name in sorted(current_ledger().researched)]
    blocked += [f"{name} (paid, awaiting age observation)" for name in current_ledger().age_up_paid]
    blocked += [
        f"{name} (paid, completion unobserved)" for name in current_ledger().research_purchases
    ]
    blocked += [
        f"{name} (paid, awaiting owned-building observation)"
        for name in current_ledger().building_purchases
    ]
    return blocked


def pending_placement_counts() -> Counter[str]:
    """Building classes awaiting wood-delta settlement, by count."""
    return Counter(p.building_class for p in current_ledger().pending_placements)


def reset_build_gates() -> None:
    """Fresh build-gate state (new game / tests)."""
    global _build_gates
    _build_gates = ActionLedger()


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
    return current_ledger().reservations().get("wood", 0)


def _rejection_reason(building_key: str, menu: str) -> str | None:
    """Human-readable reasons for the catalog's hard build restrictions."""
    cls = building_class(menu, building_key)
    if cls is None:
        return f"unknown building binding {menu}+{building_key}"
    retry_at = current_ledger().retry_after.get(f"build_{cls}", 0.0)
    if retry_at > _now():
        return f"{cls} build deferred until a fresh verification can succeed"
    suppressed_until = current_ledger().suppressed_until.get(cls, 0.0)
    if _now() < suppressed_until:
        streak = current_ledger().missing_streaks.get(cls, 0)
        return (
            f"{cls} builds suppressed for "
            f"{round(suppressed_until - _now())} more seconds: "
            f"{streak} placements in a row vanished without the wood being spent — "
            "something is systematically wrong (blocked ground, or the "
            "prerequisite isn't really standing)"
        )
    if cls == _HOUSE_CLASS and any(
        pending.building_class == _HOUSE_CLASS for pending in current_ledger().pending_placements
    ):
        return "house placement already pending HUD settlement — don't double-build"
    if cls in current_ledger().building_purchases:
        return f"{cls} was purchased and is awaiting observed construction completion"
    if cls in _UNIQUE_BUILDING_CLASSES:
        if cls in current_ledger().buildings_confirmed:
            return f"{cls} already built — one is enough; spend the wood on farms"
        if any(p.building_class == cls for p in current_ledger().pending_placements):
            return f"{cls} placement already pending wood-delta settlement — don't double-build"
    population_reading = current_ledger().population
    if cls == "house" and population_reading is not None:
        _, cap = population_reading
        if cap >= _GAME_POP_CAP_LIMIT:
            return f"house skipped: population cap {cap} is already the game maximum"
    prereqs = _BUILD_PREREQ_CLASS.get((menu, building_key), frozenset())
    missing = prereqs - current_ledger().buildings_confirmed
    if missing:
        return f"{cls} unavailable: requires completed {', '.join(sorted(missing))}"
    cost = _WOOD_COST_BY_CLASS.get(cls)
    resources = current_ledger().resources
    if cost is not None and resources is not None:
        wood = resources.get("wood")
        committed = _committed_wood()
        if wood is not None and wood - committed < cost:
            spare = wood - committed
            if committed:
                return (
                    f"{cls} unavailable: costs {cost} wood and only {spare} is uncommitted "
                    f"({wood} on the HUD, {committed} owed by placements not yet settled)"
                )
            return f"{cls} unavailable: costs {cost} wood, you have {wood}"
    if (
        menu == ECON_MENU
        and building_key in _RESOURCE_REQUIRED_KEYS
        and _resource_anchor(building_key) is None
    ):
        classes = ", ".join(sorted(_BUILD_ANCHOR_CLASSES[building_key]))
        return (
            f"{cls} skipped: no {classes} visible to build against — a drop-off camp "
            "away from its resource carries nothing; wait for the view to show one"
        )
    spec = BUILDINGS[(menu, building_key)]
    if not eligible(spec, ledger_policy_state()):
        return f"{cls} no longer eligible against the latest observation and reservations"
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
    if not current_ledger().spatial_valid:
        return ("spatial observation invalid after camera movement; refresh required", None)
    if action_dict.get("auto_placement"):
        key = str(action_dict.get("building_key", ""))
        menu = str(action_dict.get("menu") or ECON_MENU)
        placement = default_build_placement(key, menu=menu)
        if placement is None:
            # The camera moved since the pre-flight gate ran, so the resource that
            # authorised this build is no longer in frame. The trailing `h` press
            # in build_menu_steps still clears the open menu.
            return (
                f"no visible resource to anchor the {building_class(menu, key) or key} on",
                None,
            )
        return ("", placement)

    target_id = action_dict.get("target_id")
    if target_id:
        bound_revision = action_dict.get("spatial_revision")
        if bound_revision is not None and bound_revision != current_ledger().input_revision:
            return ("target observation crossed another input", None)
        expected_class = action_dict.get("expected_class")
        expected_coords = action_dict.get("expected_coords")
        for entity in _detected_entities:
            if entity.get("id") != target_id:
                continue
            if expected_class is not None and entity.get("class") != expected_class:
                return ("target class changed after selection", None)
            center = entity.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return ("target no longer has a valid center", None)
            coords = (int(center[0]), int(center[1]))
            if expected_coords is not None:
                if not isinstance(expected_coords, (list, tuple)) or len(expected_coords) != 2:
                    return ("invalid bound target coordinates", None)
                if (int(expected_coords[0]), int(expected_coords[1])) != coords:
                    return ("target moved after binding", None)
            return ("", coords)
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


def _selection_rejection(expected: str) -> str | None:
    """A navigation key is not proof that the intended command panel opened."""
    ledger = current_ledger()
    if ledger.selected_unit == expected and ledger.selected_at_revision == ledger.input_revision:
        return None
    return (
        f"{expected} selection unverified "
        f"(observed={ledger.selected_unit!r}, revision={ledger.selected_at_revision}, "
        f"input_revision={ledger.input_revision})"
    )


async def _refresh_after_input(action: str, operation_id: int) -> ActionResult:
    """Capture a post-input HUD reading while the caller still owns input."""
    if await _selection_refresh():
        return ActionResult(True, "post-input HUD captured")
    log.warning("post_input_refresh_failed", action=action, action_id=operation_id)
    return ActionResult(False, "post-input HUD refresh unavailable; operation remains pending")


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
RESCAN_SETTLE_DELAY = 0.3
DEFAULT_WAIT_MS = 100

# Ring geometry for picking open build ground around the town centre. The base
# clusters on the TC, so the emptiest ring point is almost always valid ground.
BUILD_RING_RADII: tuple[int, ...] = (280, 400, 520)
# Ring for a drop-off camp, measured from the RESOURCE. It has to land adjacent,
# so these hug far tighter than the TC ring above (a tile is ~130px at the
# deployment capture size).
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
    spec.subject: spec.price("wood") for spec in BUILDINGS.values()
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
_BUILD_PREREQ_CLASS: dict[tuple[str, str], frozenset[str]] = {
    binding: spec.requires for binding, spec in BUILDINGS.items() if spec.requires
}

# One of each is enough for this bot: a second mill/lumber camp is wasted wood
# (run 3 attempted a duplicate mill; run 6's Feudal plan re-emits the lumber
# camp build every turn and relies on this gate to stop once one stands —
# confirmed OR pending, so the settlement lag can't slip a double through).
_UNIQUE_BUILDING_CLASSES: frozenset[str] = frozenset(
    spec.subject for spec in BUILDINGS.values() if spec.unique
)

# Module-level cumulative retry telemetry (resets per process / per game).
# Surfaced via build_placement_retry log lines so the user can grep
# `total_count`/`total_seconds` to see how much turn budget got eaten by
# failed placements.

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
    entities: list[object] = list(_detected_entities)
    center = nearest_center_of_classes(entities, classes, _home_anchor())
    return None if center is None else (int(center[0]), int(center[1]))


def default_build_placement(building_key: str, *, menu: str = ECON_MENU) -> tuple[int, int] | None:
    """Screenshot-relative point to start a placement, since the text-only model
    can't see open ground and the schema carries no coordinates.

    A drop-off camp takes the emptiest point on a tight ring around its resource;
    everything else takes one around the town centre, never the TC tile itself
    (clicking on the TC always fails). None means the camp's resource is off
    screen and the caller should skip the turn.
    """
    resource = _resource_anchor(building_key) if menu == ECON_MENU else None
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
    if menu == ECON_MENU and building_key in _RESOURCE_REQUIRED_KEYS:
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
            "selection_only": True,
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
    if name in current_ledger().researched:
        return f"{name} is already researched — the HUD showed it paid for"
    if name in current_ledger().research_purchases:
        return f"{name} was paid for; completion is not directly observed"
    if name in current_ledger().age_up_paid:
        return f"{name} was paid for and is awaiting the observed age change"
    blocked_until = current_ledger().research_blocked_until.get(name, 0.0)
    if _now() < blocked_until:
        return (
            f"{name} did not take last time: the cost never left the HUD, so the "
            f"button was greyed out. Satisfy its requirement — retryable in "
            f"{round(blocked_until - _now())} seconds"
        )
    if any(p.name == name for p in current_ledger().pending_research):
        return f"{name} is already awaiting HUD settlement — don't re-press it"
    if tech.requires and tech.requires not in current_ledger().buildings_confirmed:
        return (
            f"{name} is researched at a {tech.requires} and none is confirmed "
            f"standing — build a {tech.requires} first"
        )
    resources = current_ledger().resources
    if resources is None:
        return f"{name} unavailable: no authoritative resource reading yet"
    for kind, price in _research_costs(tech).items():
        have = resources.get(kind)
        if have is not None and have < price:
            return f"{name} unavailable: costs {price} {kind}, you have {have}"
    if not eligible(RESEARCH[name], ledger_policy_state()):
        return f"{name} no longer eligible against the latest observation and reservations"
    return None


def _select_villager_step(intent: str) -> dict[str, object]:
    """Select the villager that will build, and record how (`selected_by`).

    "." is preferred: it takes an IDLE villager and re-centers the camera, so
    the placement resolves after the jump (F-33). But "." is a no-op when
    nothing is idle, and every build ends by pressing "h" — the Town Center then
    stays selected and the next "q" queues a villager instead of opening the
    menu. Run 2026_08_21_1 lost 19 of 25 placements that way.
    """
    nothing_is_idle = current_ledger().idle_present is False  # None = no reading yet
    current_ledger().selected_by = "click" if nothing_is_idle else "idle_press"
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
    building_key = action_dict.get("building_key")
    menu = str(action_dict.get("menu") or ECON_MENU)
    if isinstance(building_key, str) and (
        rejection := build_rejection(building_key, intent, menu=menu)
    ):
        return ActionResult(False, rejection)
    pending = (
        _note_pending_placement(building_key, menu=menu, point=(x, y))
        if isinstance(building_key, str)
        else None
    )
    if isinstance(building_key, str) and pending is None:
        return ActionResult(False, "fresh wood HUD baseline unavailable before placement")
    before_revision = current_ledger().input_revision
    try:
        current_ledger().note_input()
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
        if pending is not None:
            return _finish_build_placement()
    except asyncio.CancelledError:
        if pending is not None:
            current_ledger().record_interrupted_purchase(
                pending.operation_id,
                f"build_{pending.building_class}",
                before_revision,
                "cancelled during placement",
            )
        raise
    except Exception as exc:
        if pending is not None:
            current_ledger().record_interrupted_purchase(
                pending.operation_id, f"build_{pending.building_class}", before_revision, repr(exc)
            )
        raise
    return ActionResult(True, "ok")


def _finish_build_placement() -> ActionResult:
    """Release input after one click; settlement reads later observations."""
    return ActionResult(True, "placement input issued; awaiting evidence")


async def _handle_right_click(action_dict: dict[str, object], intent: str) -> ActionResult:
    fail_detail, coords = _resolve_coords(action_dict)
    if coords is None:
        log.warning("right_click_no_coords", action=action_dict)
        return ActionResult(False, fail_detail)

    x, y = coords
    if not _in_play_area(x, y):
        log.warning("right_click_off_map", x=x, y=y, intent=intent)
        return ActionResult(False, f"({x}, {y}) is in the HUD margin, not on the map")

    screen_x, screen_y = _translate(x, y)
    current_ledger().note_input()
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
    current_ledger().note_input()
    if key.lower() in CAMERA_KEYS or action_dict.get("rescan"):
        current_ledger().spatial_valid = False
    if modifiers:
        pyautogui.hotkey(*modifiers, key)
        log.info("press", key=key, modifiers=modifiers, intent=intent)
    else:
        pyautogui.press(key)
        log.info("press", key=key, intent=intent)

    # Rescan: take fresh screenshot + detection after camera-moving keys
    if action_dict.get("rescan"):
        selection_only = action_dict.get("selection_only") is True
        callback = (_selection_refresh_fn or _rescan_fn) if selection_only else _rescan_fn
        if callback is None:
            return ActionResult(False, "fresh perception is unavailable after camera movement")
        await asyncio.sleep(RESCAN_SETTLE_DELAY)
        if await callback() is False:
            return ActionResult(False, "camera moved but fresh perception did not arrive")
        if not selection_only:
            current_ledger().spatial_valid = True
        log.info("rescan_after_press", key=key, selection_only=selection_only)

    return ActionResult(True, "ok")


async def _handle_drag(action_dict: dict[str, object], intent: str) -> ActionResult:
    sx = _to_int(action_dict["start_x"])
    sy = _to_int(action_dict["start_y"])
    ex = _to_int(action_dict["end_x"])
    ey = _to_int(action_dict["end_y"])
    screen_sx, screen_sy = _translate(sx, sy)
    screen_ex, screen_ey = _translate(ex, ey)
    current_ledger().note_input()
    pyautogui.moveTo(screen_sx, screen_sy)
    pyautogui.drag(screen_ex - screen_sx, screen_ey - screen_sy, duration=0.2)
    log.info("drag", start_x=sx, start_y=sy, end_x=ex, end_y=ey, intent=intent)
    return ActionResult(True, "ok")


async def _handle_scroll(action_dict: dict[str, object], intent: str) -> ActionResult:
    clicks = _to_int(action_dict["clicks"])
    x, y = action_dict.get("x"), action_dict.get("y")
    current_ledger().note_input()
    if x is not None and y is not None:
        screen_x, screen_y = _translate(_to_int(x), _to_int(y))
        pyautogui.scroll(clicks, x=screen_x, y=screen_y)
    else:
        pyautogui.scroll(clicks)
    log.info("scroll", clicks=clicks, intent=intent)
    return ActionResult(True, "ok")


async def _handle_detect(_action_dict: dict[str, object], intent: str) -> ActionResult:
    if _rescan_full_fn:
        if await _rescan_full_fn() is False:
            return ActionResult(False, "fresh detection did not arrive")
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
    """Issue a guarded placement near the Town Center (coordinate-free).

    A click only starts settlement; the ledger must observe the purchase and
    completion before claiming that a building exists.
    """
    key = action_dict.get("building_key")
    if not isinstance(key, str) or not key:
        return ActionResult(False, "build: missing building_key")
    menu = str(action_dict.get("menu") or ECON_MENU)
    building_name = building_class(menu, key)
    action_id = f"build_{building_name}" if building_name is not None else ""
    rejection = build_rejection(key, intent, menu=menu)
    if rejection is not None:
        return ActionResult(False, rejection)
    for index, step in enumerate(build_steps(key, intent, menu=menu)):
        result = await execute_action(step)
        if not result.success:
            if index == 0 and action_id:
                current_ledger().defer_action(
                    action_id, _BUILD_REFRESH_RETRY_DELAY, "villager refresh failed"
                )
            return ActionResult(
                False, f"build failed at: {step.get('intent', '')}: {result.detail}"
            )
        if index == 0:
            if step.get("type") == "click" and not await _selection_refresh():
                if action_id:
                    current_ledger().defer_action(
                        action_id, _BUILD_REFRESH_RETRY_DELAY, "villager selection refresh failed"
                    )
                return ActionResult(False, "villager selection refresh unavailable")
            if rejection := _selection_rejection("villager"):
                if action_id:
                    current_ledger().defer_action(
                        action_id, _BUILD_REFRESH_RETRY_DELAY, "villager selection unverified"
                    )
                return ActionResult(False, rejection)
        if index == 2 and not await _selection_refresh():
            if action_id:
                current_ledger().defer_action(
                    action_id, _BUILD_REFRESH_RETRY_DELAY, "pre-placement refresh failed"
                )
            return ActionResult(False, "pre-placement HUD refresh unavailable")
    return ActionResult(True, f"placement input issued ({intent}); awaiting evidence")


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
    steps = research_steps(name, intent)
    for step in steps[:-1]:
        result = await execute_action(step)
        if not result.success:
            return ActionResult(
                False, f"research failed at: {step.get('intent', '')}: {result.detail}"
            )
    if selection_error := _selection_rejection(_TECHS[name].requires or "town_center"):
        return ActionResult(False, selection_error)
    if rejection := research_rejection(name):
        return ActionResult(False, rejection)
    pending = _note_pending_research(name, _TECHS[name])
    if pending is None:
        return ActionResult(False, "research unavailable: no resource baseline")
    before_revision = current_ledger().input_revision
    try:
        result = await execute_action(steps[-1])
    except asyncio.CancelledError:
        current_ledger().record_interrupted_purchase(
            pending.operation_id,
            _research_action_id(name),
            before_revision,
            "cancelled during research input",
        )
        raise
    if not result.success:
        current_ledger().record_interrupted_purchase(
            pending.operation_id,
            _research_action_id(name),
            before_revision,
            result.detail,
            before_spend_status="failed",
        )
        return ActionResult(False, result.detail)
    refresh = await _refresh_after_input(_research_action_id(name), pending.operation_id)
    if not refresh.success:
        return refresh
    return ActionResult(
        True,
        f"{name} pressed; the HUD spend settles it next turn — do not re-press it",
    )


async def _handle_queue_villager(action_dict: dict[str, object], intent: str) -> ActionResult:
    """Queue one villager at the TC, through the order ledger (T-531).

    Every queue path funnels here so food, housing, and prior unobserved
    production commitments cannot be bypassed by raw h+q presses.
    """
    rejection = villager_queue_rejection()
    if rejection is not None:
        return ActionResult(False, rejection)
    if not eligible(UNITS["villager"], ledger_policy_state()):
        return ActionResult(False, "villager no longer eligible against observed state")
    result = await execute_action(
        {
            "type": "press",
            "key": "h",
            "rescan": True,
            "selection_only": True,
            "intent": f"Select TC ({intent})",
        }
    )
    if not result.success:
        return ActionResult(False, f"queue_villager selection failed: {result.detail}")
    if selection_error := _selection_rejection("town_center"):
        return ActionResult(False, selection_error)
    if rejection := villager_queue_rejection():
        return ActionResult(False, rejection)
    if not eligible(UNITS["villager"], ledger_policy_state()):
        return ActionResult(False, "villager no longer eligible after selecting Town Center")
    pending = _note_pending_training("villager")
    if pending is None:
        return ActionResult(False, "villager unavailable: no resource baseline")
    before_revision = current_ledger().input_revision
    try:
        result = await execute_action(
            {"type": "press", "key": "q", "intent": f"Queue villager ({intent})"}
        )
    except asyncio.CancelledError:
        current_ledger().record_interrupted_purchase(
            pending.operation_id, "queue_villager", before_revision, "cancelled during queue input"
        )
        raise
    if not result.success:
        current_ledger().record_interrupted_purchase(
            pending.operation_id,
            "queue_villager",
            before_revision,
            result.detail,
            before_spend_status="failed",
        )
        return ActionResult(False, result.detail)
    log.info("villager_order_pending", action_id=pending.operation_id, intent=intent)
    refresh = await _refresh_after_input("queue_villager", pending.operation_id)
    if not refresh.success:
        return refresh
    return ActionResult(True, f"villager queue pending HUD settlement ({pending.operation_id})")


async def _handle_train_unit(action_dict: dict[str, object], intent: str) -> ActionResult:
    unit = action_dict.get("unit")
    if not isinstance(unit, str) or unit not in UNITS:
        return ActionResult(False, "unknown unit")
    spec = UNITS[unit]
    if unit == "villager" or not eligible(spec, ledger_policy_state()):
        return ActionResult(False, f"train_{unit} not eligible against observed state")
    result = await execute_action(
        {
            "type": "press",
            "key": spec.goto_key,
            "modifiers": list(spec.goto_modifiers),
            "rescan": True,
            "selection_only": True,
            "intent": f"Select production building for {unit}",
        }
    )
    if not result.success:
        return ActionResult(False, result.detail)
    expected_building = next(iter(spec.requires), "")
    if expected_building and (selection_error := _selection_rejection(expected_building)):
        return ActionResult(False, selection_error)
    if not eligible(spec, ledger_policy_state()):
        return ActionResult(False, f"train_{unit} no longer eligible after navigation")
    pending = _note_pending_training(unit)
    if pending is None:
        return ActionResult(False, "unit unavailable: no resource baseline")
    before_revision = current_ledger().input_revision
    try:
        result = await execute_action(
            {"type": "press", "key": spec.key, "intent": f"Train {unit} ({intent})"}
        )
    except asyncio.CancelledError:
        current_ledger().record_interrupted_purchase(
            pending.operation_id,
            f"train_{unit}",
            before_revision,
            "cancelled during training input",
        )
        raise
    if not result.success:
        current_ledger().record_interrupted_purchase(
            pending.operation_id,
            f"train_{unit}",
            before_revision,
            result.detail,
            before_spend_status="failed",
        )
        return result
    refresh = await _refresh_after_input(f"train_{unit}", pending.operation_id)
    if not refresh.success:
        return refresh
    return ActionResult(True, f"train_{unit} pressed; awaiting HUD settlement")


async def _handle_assign_idle(action_dict: dict[str, object], intent: str) -> ActionResult:
    resource = action_dict.get("resource")
    if resource not in RESOURCE_KINDS:
        return ActionResult(False, "unknown gathering resource")
    if not eligible(BY_ID[f"assign_{resource}"], ledger_policy_state()):
        return ActionResult(False, "idle assignment no longer eligible against observed state")
    result = await execute_action(
        {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager"}
    )
    if not result.success:
        current_ledger().assignment_suppressed_until[cast("ResourceKind", resource)] = (
            _now() + _ASSIGNMENT_REFRESH_RETRY_DELAY
        )
        return ActionResult(False, result.detail)
    if selection_error := _selection_rejection("villager"):
        return ActionResult(False, selection_error)
    width, height = _window_size()
    kind = cast("ResourceKind", resource)
    target = safe_gather_target(list(_detected_entities), kind, (width / 2, height / 2))
    if target is None:
        current_ledger().assignment_suppressed_until[kind] = _now() + _ASSIGNMENT_RETRY_DELAY
        log.warning(
            "assignment_target_unavailable",
            resource=kind,
            visible_classes=sorted({str(entity.get("class", "")) for entity in _detected_entities}),
            retry_seconds=_ASSIGNMENT_RETRY_DELAY,
        )
        return ActionResult(False, f"no {kind} target with verified bounds in refreshed view")
    ledger = current_ledger()
    pending = _PendingAssignment(
        operation_id=ledger.new_operation_id(),
        resource=kind,
        target_id=target.entity_id,
        idle_count_before=ledger.idle_count,
        workers_before=ledger.worker_counts.get(kind),
        noted_at_snapshot=ledger.snapshot_count,
        command_revision=ledger.input_revision + 1,
        settle_deadline=_now() + _ASSIGNMENT_SETTLE_SECONDS,
    )
    ledger.pending_assignment = pending
    ledger.record_outcome(
        ActionOutcome(pending.operation_id, f"assign_{kind}", "pending", "awaiting worker count")
    )
    log.info(
        "idle_assignment_target",
        action_id=pending.operation_id,
        resource=kind,
        target_id=target.entity_id,
        target_class=target.class_name,
        x=int(target.center[0]),
        y=int(target.center[1]),
    )
    try:
        result = await _handle_right_click(
            {
                "type": "right_click",
                "target_id": target.entity_id,
                "expected_class": target.class_name,
                "expected_coords": (int(target.center[0]), int(target.center[1])),
                "spatial_revision": ledger.input_revision,
                "intent": f"Assign idle villager to {kind} ({intent})",
            },
            intent,
        )
    except asyncio.CancelledError:
        if ledger.input_revision < pending.command_revision:
            ledger.pending_assignment = None
            ledger.record_outcome(
                ActionOutcome(pending.operation_id, f"assign_{kind}", "cancelled", "before click")
            )
        else:
            ledger.record_outcome(
                ActionOutcome(
                    pending.operation_id, f"assign_{kind}", "uncertain", "click interrupted"
                )
            )
        raise
    if not result.success:
        ledger.pending_assignment = None
        ledger.record_outcome(
            ActionOutcome(pending.operation_id, f"assign_{kind}", "failed", result.detail)
        )
        return result
    refresh = await _refresh_after_input(f"assign_{kind}", pending.operation_id)
    if not refresh.success:
        return refresh
    return ActionResult(True, "assignment pending workforce confirmation")


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
    "train_unit": _handle_train_unit,
    "assign_idle": _handle_assign_idle,
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


def as_dict(action: dict[str, object] | Action) -> dict[str, object]:
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
        preview = as_dict(action)
        if camera_moved and _uses_raw_coords_only(preview):
            log.warning("stale_coords_rejected", intent=preview.get("intent", ""))
            results.append(ActionResult(False, STALE_COORDS_DETAIL))
            continue
        result = await execute_action(action)
        results.append(result)
        if not result.success:
            break
        camera_moved = camera_moved or _moves_camera(preview)
    return results
