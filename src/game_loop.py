"""Main game loop for AoE2 LLM Agent."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import math

import structlog

from .config import config
from .executor import execute_actions, set_detected_entities, clear_detected_entities, set_rescan_fn
from .goal_logger import GoalLogger
from .goals import GoalManager
from .memory import AgentMemory, GameState
from .providers.base import BaseLLMProvider
from .providers.strategist import StrategistProvider
from .screen import capture_screenshot, save_screenshot
from .window import ensure_game_focused, get_game_window_rect, is_game_running

log = structlog.get_logger()


def _verify_actions(pre_entities: list, post_entities: list, actions: list[dict]) -> str:
    """Compare pre/post detection to infer action outcomes.

    Returns human-readable verification text for LLM context.
    """
    if not pre_entities and not post_entities:
        return ""

    results = []

    # Build lookup dicts by entity ID
    def _entity_dict(entities):
        d = {}
        for e in entities:
            eid = e.id if hasattr(e, 'id') else e.get('id', '')
            center = e.center if hasattr(e, 'center') else e.get('center', (0, 0))
            cls = e.class_name if hasattr(e, 'class_name') else e.get('class', '')
            d[eid] = {"center": center, "class": cls}
        return d

    pre_dict = _entity_dict(pre_entities)
    post_dict = _entity_dict(post_entities)

    # Check target-based actions
    for action in actions:
        target_id = action.get("target_id")
        if not target_id:
            continue

        pre = pre_dict.get(target_id)
        post = post_dict.get(target_id)
        intent = action.get("intent", "")

        if pre and not post:
            results.append(f"- {target_id}: no longer visible (moved or gathered)")
        elif pre and post:
            dx = post["center"][0] - pre["center"][0]
            dy = post["center"][1] - pre["center"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 20:
                results.append(f"- {target_id}: moved {dist:.0f}px")
            else:
                results.append(f"- {target_id}: no visible change")
        elif not pre:
            results.append(f"- {target_id}: was not detected before action")

    # Check for new entities (e.g., new building placed)
    new_ids = set(post_dict.keys()) - set(pre_dict.keys())
    if new_ids:
        new_summary = ", ".join(
            f"{eid}({post_dict[eid]['class']})" for eid in list(new_ids)[:5]
        )
        results.append(f"- New entities: {new_summary}")

    # Check for disappeared entities
    gone_ids = set(pre_dict.keys()) - set(post_dict.keys())
    if gone_ids:
        gone_summary = ", ".join(
            f"{eid}({pre_dict[eid]['class']})" for eid in list(gone_ids)[:5]
        )
        results.append(f"- Disappeared: {gone_summary}")

    return "\n".join(results) if results else ""

def _get_ground_commands(iteration: int) -> list[dict]:
    """Return hardcoded actions that execute regardless of LLM output.

    These are injected BEFORE LLM actions each turn.
    """
    commands = []
    if iteration == 1:
        # Auto Scout on turn 1 — scout explores map automatically
        commands.extend([
            {"type": "press", "key": ",", "intent": "Select scout (ground cmd)"},
            {"type": "press", "key": "g", "intent": "Auto Scout (ground cmd)"},
        ])
    return commands


# Optional detection module (graceful fallback if not available)
try:
    from detection.inference.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    log.info("detection_not_available", message="Running without YOLO detection")


async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,
    use_overlay: bool = False,
) -> AgentMemory:
    """
    Main game loop: capture → detect → alarm check → strategist → executor → act → repeat.

    Args:
        provider: LLM provider to use for decisions (text-only executor)
        max_iterations: Maximum number of iterations (None = infinite)
        memory: Optional memory instance (creates new one if not provided)
        use_detection: Whether to use YOLO detection (if available)
        time_budget: Maximum game duration in seconds (None = no limit)
        use_overlay: Whether to show live YOLO detection overlay on game window

    Returns:
        The AgentMemory instance with cumulative metrics
    """
    # Initialize memory if not provided
    if memory is None:
        memory = AgentMemory()

    # Initialize detector if available and requested
    detector = None
    if use_detection and DETECTION_AVAILABLE:
        try:
            # Use real YOLO detection (falls back to mock if model not found)
            # Prefers v5 model (latest) with configurable inference resolution
            detector = get_detector(use_mock=False, imgsz=config.detection_imgsz)
            backend = "mock" if detector.use_mock else detector.backend or "yolo"
            log.info("detector_initialized",
                     mode=backend,
                     confidence_threshold=detector.confidence_threshold)
        except Exception as e:
            log.warning("detector_init_failed", error=str(e))
            detector = None

    # Initialize detection overlay (optional)
    overlay = None
    if use_overlay:
        try:
            from .overlay import DetectionOverlay
            overlay = DetectionOverlay()
            log.info("overlay_enabled")
        except Exception as e:
            log.warning("overlay_init_failed", error=str(e))

    # Frame differencing for skipping redundant rescans
    frame_differ = None
    try:
        from detection.inference.frame_diff import FrameDiffer
        frame_differ = FrameDiffer(threshold=0.03)
    except ImportError:
        pass

    # Register rescan callback so executor can take mid-turn screenshots
    if detector:
        async def _rescan():
            if overlay:
                overlay.hide()
            # Always capture screenshot first to check if camera moved
            screenshot, _, _ = capture_screenshot(quality=50)

            # 1. Frame diff FIRST — did the camera move?
            if frame_differ and not frame_differ.has_changed(screenshot):
                # Frame unchanged — safe to use tracker prediction
                if detector.tracker and detector.tracker.get_confidence() > 0.8:
                    predicted = detector.tracker.predict()
                    set_detected_entities(predicted)
                    if overlay:
                        overlay.show(predicted, get_game_window_rect())
                    log.debug("rescan_predicted", entity_count=len(predicted))
                    return
                # Frame unchanged but tracker not confident — skip
                log.debug("rescan_skipped", reason="no_change")
                if overlay:
                    overlay.show(detector._previous_entities, get_game_window_rect())
                return

            # 2. Frame changed (camera moved) — must run actual detection
            entities = detector.detect_fast(screenshot)
            # Reset tracker if entity count dropped significantly (camera jump)
            if detector.tracker and detector._previous_entities:
                if len(entities) < len(detector._previous_entities) * 0.5:
                    detector.tracker.reset()
                    log.debug("tracker_reset", reason="camera_moved")
            set_detected_entities(entities)
            if overlay:
                overlay.show(entities, get_game_window_rect())
            log.debug("rescan_complete", entity_count=len(entities), mode="fast")
        set_rescan_fn(_rescan)

    # Initialize goal system
    goal_manager = GoalManager()
    strategist = StrategistProvider()
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    goal_logger = GoalLogger(log_dir)

    iteration = 0
    alarm = False  # Track alarm state across iterations for adaptive SAHI
    log.info("game_loop_start", provider=type(provider).__name__,
             detection=detector is not None,
             executor_model=config.model,
             strategist_model=config.strategist_model)

    # Create logs directory if saving screenshots
    screenshots_dir = None
    if config.save_screenshots:
        screenshots_dir = log_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)

    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log.info("iteration_start", iteration=iteration)

            # Check if game is running
            if not is_game_running():
                log.error("game_not_found", message="AoE2 window not found")
                break

            # Ensure game window is focused
            if not ensure_game_focused():
                log.warning("could_not_focus_game", message="Retrying in 1 second")
                await asyncio.sleep(1)
                continue

            # 1. Capture screenshot (hide overlay so it's not captured by mss)
            if overlay:
                overlay.hide()
            screenshot, width, height = capture_screenshot()
            log.debug("screenshot_captured", width=width, height=height)

            # Save screenshot if configured
            if config.save_screenshots and screenshots_dir:
                screenshot_path = screenshots_dir / f"{timestamp}_{iteration:05d}.jpg"
                save_screenshot(screenshot, str(screenshot_path))

            # 2. Run entity detection (if available)
            detected_entities = []
            if detector:
                try:
                    # Use adaptive SAHI: fast scan + targeted SAHI on entity clusters
                    if config.adaptive_sahi:
                        force_full = (
                            iteration == 1
                            or iteration % config.full_sahi_interval == 0
                            or alarm
                        )
                        detected_entities = detector.detect_adaptive(screenshot, force_full=force_full)
                    else:
                        detected_entities = detector.detect(screenshot)
                    set_detected_entities(detected_entities)
                    log.debug("detection_complete", entity_count=len(detected_entities))
                    # Show overlay with fresh detections
                    if overlay:
                        overlay.show(detected_entities, get_game_window_rect())
                except Exception as e:
                    log.warning("detection_failed", error=str(e))
                    clear_detected_entities()

            # 3. Build entity summary text for context and strategist
            entity_summary = ""
            ownership_results = {}
            if detected_entities:
                # Classify ownership of military units via color detection
                try:
                    from detection.inference.ownership import classify_entities as classify_ownership
                    from .goals import THREAT_CLASSES
                    ownership_results = classify_ownership(screenshot, detected_entities, THREAT_CLASSES)
                except Exception:
                    pass  # Non-critical — entity summary works without ownership tags

                entity_lines = []
                for entity in detected_entities[:20]:
                    eid = entity.id if hasattr(entity, 'id') else entity.get('id', 'unknown')
                    cls = entity.class_name if hasattr(entity, 'class_name') else entity.get('class', 'unknown')
                    center = entity.center if hasattr(entity, 'center') else entity.get('center', (0, 0))
                    conf = entity.confidence if hasattr(entity, 'confidence') else entity.get('confidence', 0)
                    # Add ownership tag for military units
                    owner_tag = ""
                    if eid in ownership_results:
                        owner_tag = f" [{ownership_results[eid][0].value}]"
                    entity_lines.append(f"  {eid}: {cls}{owner_tag} at ({int(center[0])},{int(center[1])}) [{conf:.0%}]")
                entity_summary = "\n".join(entity_lines)

            # 4. Alarm check — scan YOLO entities for enemy military threats
            #    Pass screenshot for color-based own/enemy classification
            alarm = goal_manager.check_alarm(detected_entities, screenshot_bytes=screenshot) if detected_entities else False

            # 5. Run strategist (every N turns OR on alarm) to create/update goals
            if strategist.should_run(iteration, alarm=alarm):
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
                             goal_count=len(new_goals),
                             alarm=alarm)
                except Exception as e:
                    log.warning("strategist_failed", error=str(e))

            # 6. Build context from memory, goals, resources, and detected entities
            context = memory.get_context_for_llm()

            # Inject cached resource readings from strategist
            resource_context = goal_manager.get_resource_context()
            if resource_context:
                context = resource_context + "\n\n" + context

            # Inject goal context
            goal_context = goal_manager.get_context_for_llm()
            if goal_context:
                context = goal_context + "\n\n" + context

            # Add detected entities to context as text for LLM
            if entity_summary:
                entity_context = "\n## Detected Entities (from YOLO)\n"
                entity_context += "Use target_class or target_id to interact with these:\n"
                entity_context += entity_summary + "\n"
                context = entity_context + "\n" + context

            # 7. Get actions from executor (text-only, no images)
            response = await provider.get_actions(context, width, height)
            reasoning = response.get("reasoning", "")
            observations = response.get("observations", {})
            actions = response.get("actions", [])

            log.info(
                "llm_response",
                iteration=iteration,
                reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
                action_count=len(actions),
            )

            # 8. Update memory with this turn
            prev_state = GameState(
                resources=dict(memory.game_state.resources),
                population=memory.game_state.population,
                population_cap=memory.game_state.population_cap,
                current_age=memory.game_state.current_age,
            )
            turn = memory.create_turn(
                reasoning=reasoning,
                actions=actions,
                observations=observations,
            )

            # 8b. Evaluate goal progress and compute reward
            goal_manager.evaluate_progress(memory.game_state, iteration)
            reward = goal_manager.compute_turn_reward(prev_state, memory.game_state)
            turn.reward = reward.get("total", 0.0)
            goal_logger.log_progress(iteration, goal_manager.active_goals, reward)

            # Log completed goals
            for goal in goal_manager.completed_goals:
                if goal.created_turn != iteration:  # Avoid logging on same turn
                    goal_logger.log_goal_completed(iteration, goal)

            if reward["total"] != 0:
                log.info("turn_reward", iteration=iteration, **reward)

            # 8c. Check for game-over via LLM observations
            game_state = observations.get("game_state", "playing") if observations else "playing"
            if game_state in ("victory", "defeat"):
                memory.game_end_reason = game_state
                log.info("game_over_detected", result=game_state, iteration=iteration)
                break

            # 8d. Check time budget
            if time_budget and memory.get_game_duration_seconds() >= time_budget:
                memory.game_end_reason = "timeout"
                log.info("time_budget_reached", seconds=time_budget, iteration=iteration)
                break

            # 9. Execute ground commands (hardcoded, before LLM actions)
            ground_cmds = _get_ground_commands(iteration)
            if ground_cmds:
                from .models import validate_actions
                ground_actions = validate_actions(ground_cmds)
                if ground_actions:
                    gc_count = await execute_actions(ground_actions)
                    log.info("ground_commands_executed", iteration=iteration,
                             count=gc_count, total=len(ground_actions))

            # 9b. Execute LLM actions
            if actions:
                success_count = await execute_actions(actions)
                memory.record_action_results(success_count, len(actions))
                log.info(
                    "actions_executed",
                    iteration=iteration,
                    total=len(actions),
                    successful=success_count,
                )

                # 9b. Post-action verification disabled for performance
                # Entity changes are picked up at the start of the next turn.
                pass
            else:
                log.warning("no_actions_fallback", iteration=iteration, reasoning=reasoning[:200])
                # Inject safe fallback: queue villager + sweep idle villagers
                fallback = [
                    {"type": "press", "key": "h", "intent": "Go to TC (fallback)"},
                    {"type": "press", "key": "q", "intent": "Queue villager (fallback)"},
                    {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager (fallback)"},
                ]
                from .models import validate_actions
                fallback_actions = validate_actions(fallback)
                if fallback_actions:
                    success_count = await execute_actions(fallback_actions)
                    memory.record_action_results(success_count, len(fallback_actions))

            # Clear detected entities after execution
            clear_detected_entities()

            # 10. Wait before next iteration
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


async def run_single_iteration(
    provider: BaseLLMProvider,
    memory: AgentMemory | None = None,
    execute: bool = False,
    use_detection: bool = True,
) -> dict:
    """
    Run a single iteration of the game loop.

    Useful for testing and debugging.

    Args:
        provider: LLM provider to use
        memory: Optional memory instance
        execute: Whether to execute actions (default False for safety)
        use_detection: Whether to use YOLO detection (if available)

    Returns:
        Dictionary with screenshot path, reasoning, observations, actions, and detected entities
    """
    if memory is None:
        memory = AgentMemory()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Capture
    screenshot, width, height = capture_screenshot()

    # Save screenshot
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = log_dir / f"test_{timestamp}.jpg"
    save_screenshot(screenshot, str(screenshot_path))

    # Run detection if available
    detected_entities = []
    if use_detection and DETECTION_AVAILABLE:
        try:
            detector = get_detector(use_mock=False)
            detected_entities = detector.detect(screenshot)
            set_detected_entities(detected_entities)
        except Exception as e:
            log.warning("detection_failed", error=str(e))

    # Build context with detected entities
    context = memory.get_context_for_llm()
    if detected_entities:
        entity_context = "\n## Detected Entities (from YOLO)\n"
        entity_context += "Use target_class or target_id to interact with these:\n"
        for entity in detected_entities[:20]:
            eid = entity.id if hasattr(entity, 'id') else entity.get('id', 'unknown')
            cls = entity.class_name if hasattr(entity, 'class_name') else entity.get('class', 'unknown')
            center = entity.center if hasattr(entity, 'center') else entity.get('center', (0, 0))
            conf = entity.confidence if hasattr(entity, 'confidence') else entity.get('confidence', 0)
            entity_context += f"  {eid}: {cls} at ({int(center[0])},{int(center[1])}) [{conf:.0%}]\n"
        context = entity_context + "\n" + context

    # Get actions (text-only, no images)
    response = await provider.get_actions(context, width, height)

    # Update memory
    memory.create_turn(
        reasoning=response.get("reasoning", ""),
        actions=response.get("actions", []),
        observations=response.get("observations", {}),
    )

    # Optionally execute
    if execute and response.get("actions"):
        await execute_actions(response["actions"])

    # Clear detection cache
    clear_detected_entities()

    return {
        "screenshot_path": str(screenshot_path),
        "reasoning": response.get("reasoning", ""),
        "observations": response.get("observations", {}),
        "actions": response.get("actions", []),
        "memory_context": context,
        "detected_entities": [
            e.to_dict() if hasattr(e, 'to_dict') else e
            for e in detected_entities
        ],
    }
