"""Main game loop for AoE2 LLM Agent."""

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from .config import config
from .entity_utils import extract_attrs, build_entity_summary
from .executor import (
    execute_actions,
    set_detected_entities,
    clear_detected_entities,
    set_rescan_fn,
    set_rescan_full_fn,
)
from .goal_logger import GoalLogger
from .goals import GoalManager
from .memory import AgentMemory, GameState
from .models import validate_actions
from .providers.base import BaseLLMProvider
from .providers.strategist import StrategistProvider
from .screen import capture_screenshot, save_screenshot
from .window import ensure_game_focused, get_game_window_rect, is_game_running

log = structlog.get_logger()

# Optional detection module (graceful fallback if not available)
try:
    from detection.inference.detector import EntityDetector, get_detector

    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    log.info("detection_not_available", message="Running without YOLO detection")


# ---------------------------------------------------------------------------
# Async detection helper
# ---------------------------------------------------------------------------

async def _invoke_detector(det: object, method: str, *args: object, **kwargs: object) -> list:
    """Call a detector method, handling both sync and async implementations."""
    fn = getattr(det, method)
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Action feedback (renamed from _verify_actions — builds text, doesn't verify)
# ---------------------------------------------------------------------------

ENTITY_MOVEMENT_THRESHOLD_PX = 20
ENTITY_DISPLAY_LIMIT = 20
RESCAN_SCREENSHOT_QUALITY = 50
TRACKER_CONFIDENCE_THRESHOLD = 0.8
ENTITY_DROP_RATIO = 0.5
FRAME_DIFFER_THRESHOLD = 0.03


def _build_action_feedback(
    pre_entities: list, post_entities: list, actions: list[dict],
) -> str:
    """Compare pre/post detection to build human-readable feedback for LLM context."""
    if not pre_entities and not post_entities:
        return ""

    def to_lookup(entities: list) -> dict:
        d = {}
        for e in entities:
            attrs = extract_attrs(e)
            d[attrs.entity_id] = {"center": attrs.center, "class": attrs.class_name}
        return d

    pre_dict = to_lookup(pre_entities)
    post_dict = to_lookup(post_entities)
    results: list[str] = []

    # Check target-based actions
    for action in actions:
        target_id = action.get("target_id")
        if not target_id:
            continue
        pre = pre_dict.get(target_id)
        post = post_dict.get(target_id)
        if pre and not post:
            results.append(f"- {target_id}: no longer visible (moved or gathered)")
        elif pre and post:
            dx = post["center"][0] - pre["center"][0]
            dy = post["center"][1] - pre["center"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > ENTITY_MOVEMENT_THRESHOLD_PX:
                results.append(f"- {target_id}: moved {dist:.0f}px")
            else:
                results.append(f"- {target_id}: no visible change")
        elif not pre:
            results.append(f"- {target_id}: was not detected before action")

    # New entities
    new_ids = set(post_dict.keys()) - set(pre_dict.keys())
    if new_ids:
        summary = ", ".join(f"{eid}({post_dict[eid]['class']})" for eid in list(new_ids)[:5])
        results.append(f"- New entities: {summary}")

    # Disappeared entities
    gone_ids = set(pre_dict.keys()) - set(post_dict.keys())
    if gone_ids:
        summary = ", ".join(f"{eid}({pre_dict[eid]['class']})" for eid in list(gone_ids)[:5])
        results.append(f"- Disappeared: {summary}")

    return "\n".join(results) if results else ""


# ---------------------------------------------------------------------------
# Ground commands
# ---------------------------------------------------------------------------

INITIAL_ZOOM_CLICKS = 5


def _get_ground_commands(iteration: int) -> list[dict]:
    """Return hardcoded actions injected BEFORE LLM actions each turn."""
    if iteration != 1:
        return []
    return [
        {"type": "scroll", "clicks": INITIAL_ZOOM_CLICKS, "intent": "Zoom in for better object detection"},
        {"type": "press", "key": ",", "intent": "Select scout (ground cmd)"},
        {"type": "press", "key": "g", "intent": "Auto Scout (ground cmd)"},
    ]


# ---------------------------------------------------------------------------
# Phase functions (extracted from game_loop body)
# ---------------------------------------------------------------------------

def _init_detector() -> object | None:
    """Initialize YOLO detector (remote or local)."""
    if not DETECTION_AVAILABLE:
        return None
    try:
        if config.detection_host:
            from detection.inference.remote_detector import get_remote_detector

            detector = get_remote_detector(config.detection_host, imgsz=config.detection_imgsz)
            log.info("detector_initialized", mode="remote", server=config.detection_host)
            return detector
        detector = get_detector(use_mock=False, imgsz=config.detection_imgsz)
        backend = "mock" if detector.use_mock else detector.backend or "yolo"
        log.info("detector_initialized", mode=backend,
                 confidence_threshold=detector.confidence_threshold)
        return detector
    except Exception as e:
        log.warning("detector_init_failed", error=str(e))
        return None


def _init_frame_differ() -> object | None:
    """Initialize frame differ for skipping redundant rescans."""
    try:
        from detection.inference.frame_diff import FrameDiffer
        return FrameDiffer(threshold=FRAME_DIFFER_THRESHOLD)
    except ImportError:
        return None


def _register_rescan_callbacks(
    detector: object, overlay: object | None, frame_differ: object | None,
) -> None:
    """Register rescan + full detection callbacks on the executor module."""

    async def _rescan() -> None:
        if overlay:
            overlay.hide()  # type: ignore[union-attr]
        screenshot, _, _ = capture_screenshot(quality=RESCAN_SCREENSHOT_QUALITY)

        # Frame diff — did the camera move?
        if frame_differ and not frame_differ.has_changed(screenshot):  # type: ignore[union-attr]
            if detector.tracker and detector.tracker.get_confidence() > TRACKER_CONFIDENCE_THRESHOLD:  # type: ignore[union-attr]
                predicted = detector.tracker.predict()  # type: ignore[union-attr]
                set_detected_entities(predicted)
                if overlay:
                    overlay.show(predicted, get_game_window_rect())  # type: ignore[union-attr]
                log.debug("rescan_predicted", entity_count=len(predicted))
                return
            log.debug("rescan_skipped", reason="no_change")
            if overlay:
                overlay.show(detector._previous_entities, get_game_window_rect())  # type: ignore[union-attr]
            return

        # Frame changed — run actual detection
        entities = await _invoke_detector(detector, "detect_fast_multi", screenshot)
        if detector.tracker and detector._previous_entities:  # type: ignore[union-attr]
            if len(entities) < len(detector._previous_entities) * ENTITY_DROP_RATIO:  # type: ignore[union-attr]
                detector.tracker.reset()  # type: ignore[union-attr]
                log.debug("tracker_reset", reason="camera_moved")
        set_detected_entities(entities)
        if overlay:
            overlay.show(entities, get_game_window_rect())  # type: ignore[union-attr]
        log.debug("rescan_complete", entity_count=len(entities), mode="fast")

    async def _rescan_full() -> None:
        if overlay:
            overlay.hide()  # type: ignore[union-attr]
        screenshot_full, _, _ = capture_screenshot(quality=85)
        if frame_differ:
            frame_differ.reset()  # type: ignore[union-attr]
        entities = await _invoke_detector(detector, "detect", screenshot_full)
        if detector.tracker:  # type: ignore[union-attr]
            detector.tracker.reset()  # type: ignore[union-attr]
        set_detected_entities(entities)
        if overlay:
            overlay.show(entities, get_game_window_rect())  # type: ignore[union-attr]
        log.info("rescan_full_complete", entity_count=len(entities))

    set_rescan_fn(_rescan)
    set_rescan_full_fn(_rescan_full)


async def _capture_screenshot(
    overlay: object | None, screenshots_dir: Path | None, iteration: int,
) -> tuple[bytes, int, int]:
    """Capture game screenshot, optionally saving to disk."""
    if overlay:
        overlay.hide()  # type: ignore[union-attr]
    screenshot, width, height = capture_screenshot()
    log.debug("screenshot_captured", width=width, height=height)

    if config.save_screenshots and screenshots_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = screenshots_dir / f"{timestamp}_{iteration:05d}.jpg"
        save_screenshot(screenshot, str(path))

    return screenshot, width, height


async def _run_detection(
    detector: object, screenshot: bytes, iteration: int, alarm: bool,
) -> list:
    """Run entity detection, choosing adaptive SAHI or standard mode."""
    try:
        if config.adaptive_sahi:
            force_full = (
                iteration == 1
                or iteration % config.full_sahi_interval == 0
                or alarm
            )
            entities = await _invoke_detector(
                detector, "detect_adaptive", screenshot, force_full=force_full,
            )
        else:
            entities = await _invoke_detector(detector, "detect", screenshot)
        set_detected_entities(entities)
        log.debug("detection_complete", entity_count=len(entities))
        return entities
    except Exception as e:
        log.warning("detection_failed", error=str(e))
        clear_detected_entities()
        return []


def _classify_entities(
    detected_entities: list, screenshot: bytes,
) -> tuple[str, dict]:
    """Build entity summary and classify ownership of military units."""
    ownership_results: dict = {}
    if not detected_entities:
        return "", ownership_results

    try:
        from detection.inference.ownership import classify_entities as classify_ownership
        from .goals import THREAT_CLASSES

        ownership_results = classify_ownership(screenshot, detected_entities, THREAT_CLASSES)
    except Exception:
        pass  # Non-critical — summary works without ownership tags

    entity_summary = build_entity_summary(
        detected_entities, max_count=ENTITY_DISPLAY_LIMIT, ownership_results=ownership_results,
    )
    return entity_summary, ownership_results


async def _run_strategist(
    strategist: StrategistProvider,
    iteration: int,
    alarm: bool,
    memory: AgentMemory,
    goal_manager: GoalManager,
    entity_summary: str,
    screenshot: bytes,
    goal_logger: GoalLogger,
) -> None:
    """Invoke the strategist to create/update goals."""
    if not strategist.should_run(iteration, alarm=alarm):
        return
    try:
        prev_goals = list(goal_manager.active_goals)
        new_goals, resource_readings = await strategist.generate_goals(
            memory.game_state,
            goal_manager.get_goals_summary(),
            entity_summary,
            iteration,
            screenshot_bytes=screenshot,
            alarm=alarm,
        )
        goal_manager.set_goals(new_goals)
        goal_manager.update_resource_readings(resource_readings, memory)
        goal_logger.log_goals_created(iteration, new_goals)
        if prev_goals:
            goal_logger.log_strategist_update(iteration, prev_goals, new_goals)
        log.info("strategist_goals_updated", turn=iteration,
                 goal_count=len(new_goals), alarm=alarm)
    except Exception as e:
        log.warning("strategist_failed", error=str(e))


def _build_llm_context(
    memory: AgentMemory,
    goal_manager: GoalManager,
    entity_summary: str,
) -> str:
    """Assemble the full text context for the executor LLM."""
    context = memory.get_context_for_llm()

    resource_context = goal_manager.get_resource_context()
    if resource_context:
        context = resource_context + "\n\n" + context

    goal_context = goal_manager.get_context_for_llm()
    if goal_context:
        context = goal_context + "\n\n" + context

    if entity_summary:
        entity_context = (
            "\n## Detected Entities (from YOLO)\n"
            "Use target_class or target_id to interact with these:\n"
            + entity_summary + "\n"
        )
        context = entity_context + "\n" + context

    return context


def _process_response(
    response: dict,
    memory: AgentMemory,
    goal_manager: GoalManager,
    iteration: int,
    goal_logger: GoalLogger,
    time_budget: float | None,
) -> tuple[list, str | None]:
    """Parse LLM response, update memory/goals, check for game-over.

    Returns (actions, game_end_reason). game_end_reason is None if game continues.
    """
    reasoning = response.get("reasoning", "")
    observations = response.get("observations", {})
    actions = response.get("actions", [])

    log.info(
        "llm_response",
        iteration=iteration,
        reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        action_count=len(actions),
    )

    # Snapshot previous state for reward computation
    prev_state = GameState(
        resources=dict(memory.game_state.resources),
        population=memory.game_state.population,
        population_cap=memory.game_state.population_cap,
        current_age=memory.game_state.current_age,
    )
    turn = memory.create_turn(reasoning=reasoning, actions=actions, observations=observations)

    # Evaluate goals and reward
    goal_manager.evaluate_progress(memory.game_state, iteration)
    reward = goal_manager.compute_turn_reward(prev_state, memory.game_state)
    turn.reward = reward.get("total", 0.0)
    goal_logger.log_progress(iteration, goal_manager.active_goals, reward)

    for goal in goal_manager.completed_goals:
        if goal.created_turn != iteration:
            goal_logger.log_goal_completed(iteration, goal)

    if reward["total"] != 0:
        log.info("turn_reward", iteration=iteration, **reward)

    # Game-over checks
    game_state = observations.get("game_state", "playing") if observations else "playing"
    if game_state in ("victory", "defeat"):
        memory.game_end_reason = game_state
        log.info("game_over_detected", result=game_state, iteration=iteration)
        return actions, game_state

    if time_budget and memory.get_game_duration_seconds() >= time_budget:
        memory.game_end_reason = "timeout"
        log.info("time_budget_reached", seconds=time_budget, iteration=iteration)
        return actions, "timeout"

    return actions, None


async def _execute_turn_actions(
    actions: list, iteration: int, memory: AgentMemory, reasoning: str,
) -> None:
    """Execute ground commands, LLM actions, or fallback actions."""
    # Ground commands (hardcoded, before LLM actions)
    ground_cmds = _get_ground_commands(iteration)
    if ground_cmds:
        ground_actions = validate_actions(ground_cmds)
        if ground_actions:
            gc_results = await execute_actions(ground_actions)
            gc_count = sum(1 for r in gc_results if r.success)
            log.info("ground_commands_executed", iteration=iteration,
                     count=gc_count, total=len(ground_actions))

    # LLM actions
    if actions:
        results = await execute_actions(actions)
        success_count = sum(1 for r in results if r.success)
        memory.record_action_results(success_count, len(actions))
        log.info("actions_executed", iteration=iteration,
                 total=len(actions), successful=success_count)

        # Build feedback from failed actions
        verification_lines = []
        for action, result in zip(actions, results):
            if not result.success:
                a_intent = action.get("intent", "") if isinstance(action, dict) else ""
                a_type = action.get("type", "") if isinstance(action, dict) else ""
                verification_lines.append(f"- FAILED {a_type}: {a_intent} — {result.detail}")
        if verification_lines:
            memory.set_last_verification("\n".join(verification_lines))
    else:
        log.warning("no_actions_fallback", iteration=iteration, reasoning=reasoning[:200])
        fallback = [
            {"type": "press", "key": "h", "intent": "Go to TC (fallback)"},
            {"type": "press", "key": "q", "intent": "Queue villager (fallback)"},
            {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager (fallback)"},
        ]
        fallback_actions = validate_actions(fallback)
        if fallback_actions:
            fb_results = await execute_actions(fallback_actions)
            fb_success = sum(1 for r in fb_results if r.success)
            memory.record_action_results(fb_success, len(fallback_actions))

    clear_detected_entities()


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
    log.info("game_loop_start", provider=type(provider).__name__,
             detection=detector is not None,
             executor_model=config.model,
             strategist_model=config.strategist_model)

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
                overlay, screenshots_dir, iteration,
            )

            detected_entities = []
            if detector:
                detected_entities = await _run_detection(
                    detector, screenshot, iteration, alarm,
                )
                if overlay and detected_entities:
                    overlay.show(detected_entities, get_game_window_rect())

            entity_summary, _ownership = _classify_entities(detected_entities, screenshot)

            alarm = (
                goal_manager.check_alarm(detected_entities, screenshot_bytes=screenshot)
                if detected_entities
                else False
            )

            await _run_strategist(
                strategist, iteration, alarm, memory,
                goal_manager, entity_summary, screenshot, goal_logger,
            )

            context = _build_llm_context(memory, goal_manager, entity_summary)

            response = await provider.get_actions(context, width, height)

            actions, game_end_reason = _process_response(
                response, memory, goal_manager, iteration, goal_logger, time_budget,
            )
            if game_end_reason:
                break

            await _execute_turn_actions(
                actions, iteration, memory, response.get("reasoning", ""),
            )

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot, width, height = capture_screenshot()

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = log_dir / f"test_{timestamp}.jpg"
    save_screenshot(screenshot, str(screenshot_path))

    detected_entities: list = []
    if use_detection and DETECTION_AVAILABLE:
        try:
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
            "Use target_class or target_id to interact with these:\n"
            + summary + "\n"
        )
        context = entity_context + "\n" + context

    response = await provider.get_actions(context, width, height)

    memory.create_turn(
        reasoning=response.get("reasoning", ""),
        actions=response.get("actions", []),
        observations=response.get("observations", {}),
    )

    if execute and response.get("actions"):
        await execute_actions(response["actions"])

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
