"""Main game loop for AoE2 LLM Agent."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .providers.base import LLMResult
    from .providers.claude import ClaudeProvider

from . import reactive
from .config import config
from .detection_phase import (
    DETECTION_AVAILABLE,
    ENTITY_DISPLAY_LIMIT,
    _capture_screenshot,
    _classify_entities,
    _init_detector,
    _init_frame_differ,
    _register_rescan_callbacks,
    _run_detection,
)
from .entity_utils import build_entity_summary
from .executor import (
    can_resolve,
    clear_detected_entities,
    confirmed_buildings,
    execute_actions,
    observe_hud,
    reset_build_gates,
    set_detected_entities,
    villagers_ordered,
)
from .goal_logger import GoalLogger
from .goals import GoalManager
from .memory import AgentMemory
from .models import validate_actions
from .providers.strategist import StrategistProvider, get_default_goals, read_hud_readings
from .resource_ocr import warm_up_ocr
from .screen import capture_screenshot, save_screenshot
from .strategist_phase import _maybe_launch_strategist
from .turn_phases import (
    _build_llm_context,
    _execute_turn_actions,
    _get_ground_commands,
    _process_response,
    known_buildings_line,
)
from .window import ensure_game_focused, get_game_window_rect, is_game_running

log = structlog.stdlib.get_logger()

# Consecutive focus failures (≈2 s each: 3 in-window retries + 1 s sleep) before
# the run aborts with game_end_reason="lost_focus". Run 1 burned 12 of 30
# iterations retrying an unfocusable window with no end-reason label (F-1).
_MAX_FOCUS_FAILURES = 15


# ---------------------------------------------------------------------------
# Turn pipelining (S6, RTC-style)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    """A single-shot plan launched last turn, executed this turn (RTC overlap)."""

    task: asyncio.Task[LLMResult]
    iteration: int


def _should_pipeline(context: str, provider: ClaudeProvider) -> bool:
    """Whether this turn is routine (single-shot) and can pipeline.

    Only single-shot (routine) turns pipeline: the tool loop executes its
    actions during its own run, so pre-launching it would double-act. Combat
    and housing emergencies return False and run synchronously.
    """
    return provider._use_single_shot(context)


def _commit_head(actions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep only the leading actions; the tail is discarded because next turn's
    plan supersedes it from fresher perception."""
    return actions[: config.pipeline_commit_max]


def _revalidate_against_fresh(actions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop committed actions whose target no longer resolves against the current
    entity cache — a plan computed last turn may have gone stale."""
    return [action for action in actions if can_resolve(action)]


async def _execute_or_record(
    response: LLMResult,
    actions: list[dict[str, object]],
    memory: AgentMemory,
    iteration: int,
) -> None:
    """Execute the actions, or only record results when the provider's tool loop
    already executed them this turn."""
    if response.get("actions_already_executed"):
        success = response.get("success_count", len(actions))
        memory.record_action_results(success, len(actions))
        log.info("actions_executed", iteration=iteration, total=len(actions), successful=success)
    else:
        await _execute_turn_actions(actions, iteration, memory, response.get("reasoning", ""))


async def _run_routine_upkeep(
    iteration: int,
    memory: AgentMemory,
    detected_entities: list[object],
    alarm: bool,
) -> None:
    """Routine villager upkeep — the work done in the LLM's in-flight window
    while the turn's plan computes. (Opening ground commands run earlier, before
    the first perception pass — see the iteration-1 block in `game_loop`.)"""
    routine_cmds = reactive.decide(detected_entities, memory.game_state, alarm)
    if routine_cmds:
        routine_actions = validate_actions(routine_cmds)
        if routine_actions:
            await execute_actions(routine_actions)
            log.info("routine_executed", iteration=iteration, count=len(routine_actions))


def _register_focus_failure(memory: AgentMemory, failures: int) -> tuple[int, bool]:
    """Count one focus failure; True means give up (lost_focus recorded)."""
    failures += 1
    if failures >= _MAX_FOCUS_FAILURES:
        log.error("focus_lost_giving_up", failures=failures)
        memory.game_end_reason = "lost_focus"
        return failures, True
    log.warning("could_not_focus_game", message="Retrying in 1 second", failures=failures)
    return failures, False


def _sync_turn_state(
    memory: AgentMemory,
    goal_manager: GoalManager,
    hud_readings: dict,
) -> None:
    """Once-per-iteration state upkeep from this turn's HUD reading.

    Deliberately NOT in update_from_observations (which fires more than once
    per turn): the idle-badge streak feeds the reactive tier's count trust
    gate, the buildings-seen evidence feeds its Feudal prep/age-up gating, and
    observe_hud feeds the executor's build gates (house headroom,
    prerequisites, wood cost, placement settlement).
    """
    if hud_readings:
        goal_manager.update_resource_readings(hud_readings, memory)
    game_state = memory.game_state
    game_state.idle_streak = (game_state.idle_streak + 1) if game_state.idle_present else 0
    game_state.buildings_seen = confirmed_buildings()
    game_state.villagers_ordered = villagers_ordered()
    observe_hud(game_state.population, game_state.population_cap, game_state.resources)


async def _drain_pending(
    plan: _PendingPlan,
    memory: AgentMemory,
    goal_manager: GoalManager,
    iteration: int,
    goal_logger: GoalLogger,
    time_budget: float | None,
) -> str | None:
    """Await last turn's plan and execute its revalidated head. Returns a
    game_end_reason when the game ended, else None."""
    response = await plan.task
    actions, game_end_reason = _process_response(
        response, memory, goal_manager, iteration, goal_logger, time_budget
    )
    if game_end_reason:
        return game_end_reason
    head = _commit_head(_revalidate_against_fresh(actions))
    log.info(
        "pipeline_head_committed",
        iteration=iteration,
        from_iteration=plan.iteration,
        committed=len(head),
        proposed=len(actions),
    )
    await _execute_or_record(response, head, memory, iteration)
    return None


async def _cancel_pending(plan: _PendingPlan | None) -> None:
    """Discard an in-flight plan (combat transition or shutdown)."""
    if plan is None or plan.task.done():
        return
    plan.task.cancel()
    # CancelledError is a BaseException (Python 3.8+) — suppress(Exception)
    # alone would let the cancellation we just requested escape the caller.
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await plan.task


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


async def game_loop(
    provider: ClaudeProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,
    use_overlay: bool = False,
) -> AgentMemory:
    """Main game loop: capture -> detect -> strategist -> execute -> repeat."""
    if memory is None:
        memory = AgentMemory()
    reset_build_gates()  # per-game state: HUD snapshot + buildings seen

    # Initialize subsystems
    detector = _init_detector() if use_detection else None

    # Build the OCR engine now, off the loop, instead of lazily inside the first
    # turn — engine construction + first inference cost ~10-15 s on the VM and were
    # the bulk of the post-startup freeze (2026-07-11 run review, F-6). Only the
    # rapidocr backend uses the engine (autodetect aside, which implies rapidocr
    # setups in practice). The task reference is kept so it isn't GC'd mid-warm-up;
    # warm_up_ocr never raises.
    ocr_warmup_task: asyncio.Task[None] | None = None
    if config.ocr_backend == "rapidocr":
        ocr_warmup_task = asyncio.create_task(asyncio.to_thread(warm_up_ocr))

    overlay = None
    if use_overlay:
        try:
            from .overlay import DetectionOverlay

            overlay = DetectionOverlay()
            log.info("overlay_enabled")
        except Exception as e:
            log.warning("overlay_init_failed", error=str(e))

    frame_differ = _init_frame_differ()

    if detector:
        _register_rescan_callbacks(detector, overlay, frame_differ)

    # Initialize goal system
    goal_manager = GoalManager()
    goal_manager.set_goals(get_default_goals(turn=0))
    strategist = StrategistProvider()
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    goal_logger = GoalLogger(log_dir)

    screenshots_dir = None
    if config.save_screenshots:
        screenshots_dir = log_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)

    iteration = 0
    alarm = False
    focus_failures = 0  # consecutive — reset on every successful focus
    strategist_task: asyncio.Task | None = None
    pending_plan: _PendingPlan | None = None  # S6: plan from last turn, run this turn

    # Propagate cross-game memory titles from the provider onto memory so
    # _process_response can validate `[applied: ...]` prefixes against the set
    # of titles that were actually injected into the system prompt this game.
    memory.memories_loaded = list(getattr(provider, "loaded_memory_titles", []))
    log.info("memories_loaded", count=len(memory.memories_loaded), titles=memory.memories_loaded)

    log.info(
        "game_loop_start",
        provider=type(provider).__name__,
        detection=detector is not None,
        executor_model=config.model,
        strategist_model=config.strategist_model,
    )

    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            log.info("iteration_start", iteration=iteration)

            if not is_game_running():
                log.error("game_not_found", message="AoE2 window not found")
                if not memory.game_end_reason:
                    memory.game_end_reason = "game_not_found"
                break
            if not ensure_game_focused():
                # Unplayable time isn't billed against the iteration budget —
                # this attempt is retried under the same iteration number.
                iteration -= 1
                focus_failures, give_up = _register_focus_failure(memory, focus_failures)
                if give_up:
                    break
                await asyncio.sleep(1)
                continue
            focus_failures = 0

            if iteration == 1:
                # The opening (zoom, select scout, auto-scout) needs no perception —
                # run it before the first OCR/detection pass, which is the slowest
                # of the game (engine warm-up), so the scout explores during that
                # wait instead of after it.
                ground_actions = validate_actions(_get_ground_commands(iteration))
                if ground_actions:
                    await execute_actions(ground_actions)

            screenshot, width, height = await _capture_screenshot(
                overlay,
                screenshots_dir,
                iteration,
            )

            # Refresh game_state from the resource bar EVERY turn via local OCR,
            # independent of the slow periodic strategist. Keeping population /
            # resources current is what makes the housed/pop/resource signals
            # accurate for the alarm check, strategist, and executor — so e.g. the
            # build-house path triggers the turn the agent actually gets housed.
            hud_readings, calib = await read_hud_readings(screenshot, turn=iteration)
            _sync_turn_state(memory, goal_manager, hud_readings)
            # Show the OCR reading regions on the debug overlay (--overlay only).
            if overlay is not None and calib is not None:
                overlay.set_ocr_fields(calib.field_rects())

            detected_entities = []
            if detector:
                detected_entities = await _run_detection(
                    detector,
                    screenshot,
                    iteration,
                    alarm,
                )
            # Render every turn the overlay is enabled (even with no detections) so
            # the resource-bar OCR boxes stay visible.
            if overlay is not None:
                overlay.show(detected_entities, get_game_window_rect())

            entity_summary, _ownership = _classify_entities(detected_entities, screenshot)

            alarm = (
                goal_manager.check_alarm(detected_entities, screenshot_bytes=screenshot)
                if detected_entities
                else False
            )

            strategist_task = _maybe_launch_strategist(
                strategist,
                iteration,
                alarm,
                memory,
                goal_manager,
                entity_summary,
                hud_readings,
                known_buildings_line(detected_entities),
                goal_logger,
                strategist_task,
            )

            context = _build_llm_context(memory, goal_manager, entity_summary, detected_entities)

            # S6: pipeline routine turns — compute this turn's plan while last
            # turn's committed head executes; combat turns stay synchronous.
            if _should_pipeline(context, provider):
                this_task = asyncio.create_task(provider.get_actions(context, width, height))
                await _run_routine_upkeep(iteration, memory, detected_entities, alarm)
                if pending_plan is not None:
                    game_end_reason = await _drain_pending(
                        pending_plan, memory, goal_manager, iteration, goal_logger, time_budget
                    )
                    if game_end_reason:
                        break
                pending_plan = _PendingPlan(task=this_task, iteration=iteration)
            else:
                # Combat/tool-loop turn: discard the stale routine plan and
                # act synchronously on the current frame.
                await _cancel_pending(pending_plan)
                pending_plan = None
                await _run_routine_upkeep(iteration, memory, detected_entities, alarm)
                response = await provider.get_actions(context, width, height)
                actions, game_end_reason = _process_response(
                    response, memory, goal_manager, iteration, goal_logger, time_budget
                )
                if game_end_reason:
                    break
                await _execute_or_record(response, actions, memory, iteration)

            await asyncio.sleep(config.loop_delay)

        # Normal exit: either max_iterations ran out, or a break above set the
        # reason (victory/defeat/timeout/game_not_found). Never leave it empty —
        # an unlabeled run is indistinguishable from a truncated one in results.tsv.
        if not memory.game_end_reason:
            memory.game_end_reason = "iterations_exhausted"

    except KeyboardInterrupt:
        log.info("game_loop_interrupted", iterations=iteration)
        if not memory.game_end_reason:
            memory.game_end_reason = "interrupted"
    except Exception as e:
        log.error("game_loop_error", error=str(e), iteration=iteration)
        if not memory.game_end_reason:
            memory.game_end_reason = "error"
        raise
    finally:
        # Await any in-flight strategist task so it can update goals/log cleanly.
        # Errors are already logged inside _run_strategist_async.
        if strategist_task is not None and not strategist_task.done():
            with contextlib.suppress(Exception):
                await strategist_task
        # Discard any pipelined plan still in flight (S6). The warm-up thread
        # itself can't be interrupted — cancelling just stops waiting on it.
        await _cancel_pending(pending_plan)
        if ocr_warmup_task is not None and not ocr_warmup_task.done():
            ocr_warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ocr_warmup_task
        if overlay:
            overlay.close()
        # Exits that bypass both except clauses (e.g. CancelledError, a
        # BaseException) reach here with no reason set — run 13 logged
        # game_end_reason="" on a manual stop (T-543). This is the one choke
        # point every exit passes through, so enforce the invariant here.
        if not memory.game_end_reason:
            memory.game_end_reason = "interrupted"
        metrics = memory.get_metrics_snapshot()
        log.info("game_metrics_final", **metrics)
        goal_logger.log_game_end(
            iteration,
            memory.game_end_reason,
            len(goal_manager.completed_goals),
        )

    return memory


# ---------------------------------------------------------------------------
# Single iteration (testing / debugging)
# ---------------------------------------------------------------------------


async def run_single_iteration(
    provider: ClaudeProvider,
    memory: AgentMemory | None = None,
    execute: bool = False,
    use_detection: bool = True,
) -> dict:
    """Run a single iteration of the game loop for testing."""
    if memory is None:
        memory = AgentMemory()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    screenshot, width, height = capture_screenshot()

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = log_dir / f"test_{timestamp}.jpg"
    save_screenshot(screenshot, str(screenshot_path))

    detected_entities: list = []
    if use_detection and DETECTION_AVAILABLE:
        try:
            from detection.inference.detector import get_detector

            detector = get_detector(use_mock=False, model_name=config.detection_model)
            detected_entities = detector.detect(screenshot)
            set_detected_entities(detected_entities)
        except Exception as e:
            log.warning("detection_failed", error=str(e))

    context = memory.get_context_for_llm()
    if detected_entities:
        summary = build_entity_summary(detected_entities, max_count=ENTITY_DISPLAY_LIMIT)
        entity_context = (
            "\n## Detected Entities (from YOLO)\n"
            "Use target_class or target_id to interact with these:\n" + summary + "\n"
        )
        context = entity_context + "\n" + context

    response = await provider.get_actions(context, width, height)

    memory.create_turn(
        reasoning=response.get("reasoning", ""),
        actions=response.get("actions", []),
        observations=response.get("observations", {}),
    )

    actions = response.get("actions") or []
    if execute and actions:
        await execute_actions(actions)

    clear_detected_entities()

    return {
        "screenshot_path": str(screenshot_path),
        "reasoning": response.get("reasoning", ""),
        "observations": response.get("observations", {}),
        "actions": response.get("actions", []),
        "memory_context": context,
        "detected_entities": [
            e.to_dict() if hasattr(e, "to_dict") else e for e in detected_entities
        ],
    }
