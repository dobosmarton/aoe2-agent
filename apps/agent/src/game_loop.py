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
    from .providers.base import BaseLLMProvider, LLMResult

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
    execute_actions,
    set_detected_entities,
)
from .goal_logger import GoalLogger
from .goals import GoalManager
from .memory import AgentMemory
from .models import validate_actions
from .providers.claude import ClaudeProvider
from .providers.strategist import StrategistProvider, get_default_goals
from .screen import capture_screenshot, save_screenshot
from .strategist_phase import _maybe_launch_strategist
from .turn_phases import (
    _build_llm_context,
    _execute_turn_actions,
    _get_ground_commands,
    _process_response,
)
from .window import ensure_game_focused, get_game_window_rect, is_game_running

log = structlog.stdlib.get_logger()


# ---------------------------------------------------------------------------
# Turn pipelining (S6, RTC-style)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    """A single-shot plan launched last turn, executed this turn (RTC overlap)."""

    task: asyncio.Task[LLMResult]
    iteration: int


def _should_pipeline(context: str, provider: BaseLLMProvider) -> bool:
    """Whether this turn is routine (single-shot) and can pipeline.

    Only single-shot (routine) turns pipeline: the tool loop executes its
    actions during its own run, so pre-launching it would double-act. Combat
    and housing emergencies return False and run synchronously.
    """
    if isinstance(provider, ClaudeProvider):
        return provider._use_single_shot(context)
    return False


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
    """Ground commands + routine villager upkeep — the work done in the LLM's
    in-flight window while the turn's plan computes."""
    ground_cmds = _get_ground_commands(iteration)
    if ground_cmds:
        ground_actions = validate_actions(ground_cmds)
        if ground_actions:
            await execute_actions(ground_actions)
    routine_cmds = reactive.decide(detected_entities, memory.game_state, alarm)
    if routine_cmds:
        routine_actions = validate_actions(routine_cmds)
        if routine_actions:
            await execute_actions(routine_actions)
            log.info("routine_executed", iteration=iteration, count=len(routine_actions))


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
    with contextlib.suppress(Exception):
        await plan.task


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,
    use_overlay: bool = False,
) -> AgentMemory:
    """Main game loop: capture -> detect -> strategist -> execute -> repeat."""
    if memory is None:
        memory = AgentMemory()

    # Initialize subsystems
    detector = _init_detector() if use_detection else None

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
                break
            if not ensure_game_focused():
                log.warning("could_not_focus_game", message="Retrying in 1 second")
                await asyncio.sleep(1)
                continue

            screenshot, width, height = await _capture_screenshot(
                overlay,
                screenshots_dir,
                iteration,
            )

            detected_entities = []
            if detector:
                detected_entities = await _run_detection(
                    detector,
                    screenshot,
                    iteration,
                    alarm,
                )
                if overlay and detected_entities:
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
                screenshot,
                goal_logger,
                strategist_task,
            )

            context = _build_llm_context(memory, goal_manager, entity_summary)

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
        # Discard any pipelined plan still in flight (S6).
        await _cancel_pending(pending_plan)
        if overlay:
            overlay.close()
        metrics = memory.get_metrics_snapshot()
        log.info("game_metrics_final", **metrics)
        goal_logger.log_game_end(
            iteration,
            memory.game_end_reason or "unknown",
            len(goal_manager.completed_goals),
        )

    return memory


# ---------------------------------------------------------------------------
# Single iteration (testing / debugging)
# ---------------------------------------------------------------------------


async def run_single_iteration(
    provider: BaseLLMProvider,
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

            detector = get_detector(use_mock=False)
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
