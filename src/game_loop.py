"""Main game loop for AoE2 LLM Agent."""

import asyncio
from datetime import datetime
from pathlib import Path

import structlog

from .config import config
from .executor import execute_actions
from .memory import AgentMemory
from .providers.base import BaseLLMProvider
from .screen import capture_screenshot, save_screenshot
from .window import ensure_game_focused, is_game_running

log = structlog.get_logger()


async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
) -> None:
    """
    Main game loop: capture → think → act → repeat.

    Args:
        provider: LLM provider to use for decisions
        max_iterations: Maximum number of iterations (None = infinite)
        memory: Optional memory instance (creates new one if not provided)
    """
    # Initialize memory if not provided
    if memory is None:
        memory = AgentMemory()

    iteration = 0
    log.info("game_loop_start", provider=type(provider).__name__)

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

            # 2. Build context from memory
            context = memory.get_context_for_llm()

            # 3. Get actions from LLM
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

            # 4. Update memory with this turn
            memory.create_turn(
                reasoning=reasoning,
                actions=actions,
                observations=observations,
            )

            # 5. Execute actions
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

            # 6. Wait before next iteration
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
) -> dict:
    """
    Run a single iteration of the game loop.

    Useful for testing and debugging.

    Args:
        provider: LLM provider to use
        memory: Optional memory instance
        execute: Whether to execute actions (default False for safety)

    Returns:
        Dictionary with screenshot path, reasoning, observations, and actions
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

    # Build context
    context = memory.get_context_for_llm()

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

    return {
        "screenshot_path": str(screenshot_path),
        "reasoning": response.get("reasoning", ""),
        "observations": response.get("observations", {}),
        "actions": response.get("actions", []),
        "memory_context": context,
    }
