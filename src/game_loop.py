"""Main game loop for AoE2 LLM Agent."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from .config import config
from .executor import execute_actions, set_detected_entities, clear_detected_entities
from .memory import AgentMemory
from .providers.base import BaseLLMProvider
from .screen import capture_screenshot, save_screenshot
from .window import ensure_game_focused, is_game_running

log = structlog.get_logger()

# Optional detection module (graceful fallback if not available)
try:
    from detection.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    log.info("detection_not_available", message="Running without YOLO detection")


async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
) -> None:
    """
    Main game loop: capture → detect → think → act → repeat.

    Args:
        provider: LLM provider to use for decisions
        max_iterations: Maximum number of iterations (None = infinite)
        memory: Optional memory instance (creates new one if not provided)
        use_detection: Whether to use YOLO detection (if available)
    """
    # Initialize memory if not provided
    if memory is None:
        memory = AgentMemory()

    # Initialize detector if available and requested
    detector = None
    if use_detection and DETECTION_AVAILABLE:
        try:
            # Use mock mode if model not available
            detector = get_detector(use_mock=True)
            log.info("detector_initialized", mode="mock" if detector.use_mock else "yolo")
        except Exception as e:
            log.warning("detector_init_failed", error=str(e))
            detector = None

    iteration = 0
    log.info("game_loop_start", provider=type(provider).__name__,
             detection=detector is not None)

    # Create logs directory if saving screenshots
    screenshots_dir = None
    if config.save_screenshots:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
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

            # 1. Capture screenshot
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
                    detected_entities = detector.detect(screenshot)
                    set_detected_entities(detected_entities)
                    log.debug("detection_complete", entity_count=len(detected_entities))
                except Exception as e:
                    log.warning("detection_failed", error=str(e))
                    clear_detected_entities()

            # 3. Build context from memory and detected entities
            context = memory.get_context_for_llm()

            # Add detected entities to context for LLM
            if detected_entities:
                entity_context = "\n## Detected Entities\n"
                for entity in detected_entities[:15]:  # Limit to avoid token bloat
                    eid = entity.id if hasattr(entity, 'id') else entity.get('id', 'unknown')
                    cls = entity.class_name if hasattr(entity, 'class_name') else entity.get('class', 'unknown')
                    center = entity.center if hasattr(entity, 'center') else entity.get('center', (0, 0))
                    conf = entity.confidence if hasattr(entity, 'confidence') else entity.get('confidence', 0)
                    entity_context += f"  {eid}: {cls} at ({int(center[0])},{int(center[1])}) [{conf:.0%}]\n"
                context = entity_context + "\n" + context

            # 4. Get actions from LLM
            response = await provider.get_actions(screenshot, context, width, height)
            reasoning = response.get("reasoning", "")
            observations = response.get("observations", {})
            actions = response.get("actions", [])

            log.info(
                "llm_response",
                iteration=iteration,
                reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
                action_count=len(actions),
            )

            # 5. Update memory with this turn
            memory.create_turn(
                reasoning=reasoning,
                actions=actions,
                observations=observations,
            )

            # 6. Execute actions
            if actions:
                success_count = await execute_actions(actions)
                log.info(
                    "actions_executed",
                    iteration=iteration,
                    total=len(actions),
                    successful=success_count,
                )
            else:
                log.warning("no_actions", iteration=iteration, reasoning=reasoning[:200])

            # Clear detected entities after execution
            clear_detected_entities()

            # 7. Wait before next iteration
            await asyncio.sleep(config.loop_delay)

    except KeyboardInterrupt:
        log.info("game_loop_interrupted", iterations=iteration)
    except Exception as e:
        log.error("game_loop_error", error=str(e), iteration=iteration)
        raise


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
            detector = get_detector(use_mock=True)
            detected_entities = detector.detect(screenshot)
            set_detected_entities(detected_entities)
        except Exception as e:
            log.warning("detection_failed", error=str(e))

    # Build context with detected entities
    context = memory.get_context_for_llm()
    if detected_entities:
        entity_context = "\n## Detected Entities\n"
        for entity in detected_entities[:15]:
            eid = entity.id if hasattr(entity, 'id') else entity.get('id', 'unknown')
            cls = entity.class_name if hasattr(entity, 'class_name') else entity.get('class', 'unknown')
            center = entity.center if hasattr(entity, 'center') else entity.get('center', (0, 0))
            conf = entity.confidence if hasattr(entity, 'confidence') else entity.get('confidence', 0)
            entity_context += f"  {eid}: {cls} at ({int(center[0])},{int(center[1])}) [{conf:.0%}]\n"
        context = entity_context + "\n" + context

    # Get actions
    response = await provider.get_actions(screenshot, context, width, height)

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
