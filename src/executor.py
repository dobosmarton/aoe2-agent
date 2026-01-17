"""Action executor module for AoE2 LLM Agent."""

import asyncio
from typing import Any

import pyautogui
import structlog

from .config import config
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.get_logger()

# Configure pyautogui for better game compatibility
pyautogui.FAILSAFE = False  # Disable failsafe (mouse to corner stops execution)
pyautogui.PAUSE = 0.02  # Reduce default pause between actions (default is 0.1)

# Cache window position (updated before action batches)
_window_offset: tuple[int, int] = (0, 0)


async def execute_action(action: dict[str, Any] | Action) -> bool:
    """
    Execute a single action from LLM output.

    Args:
        action: Action dictionary or validated Action model

    Returns:
        True if action executed successfully, False otherwise
    """
    # Convert to dict if it's a Pydantic model
    if hasattr(action, "model_dump"):
        action_dict = action.model_dump()
    else:
        action_dict = action

    # Validate action if not already validated
    if not hasattr(action, "model_dump"):
        validated = validate_action(action_dict)
        if not validated:
            log.warning("invalid_action", action=action_dict)
            return False
        action_dict = validated.model_dump()

    action_type = action_dict.get("type")
    intent = action_dict.get("intent", "")

    try:
        # Translate coordinates from screenshot-relative to screen-absolute
        def translate(x: int, y: int) -> tuple[int, int]:
            return (x + _window_offset[0], y + _window_offset[1])

        if action_type == "click":
            x, y = action_dict["x"], action_dict["y"]
            screen_x, screen_y = translate(x, y)
            pyautogui.click(screen_x, screen_y)
            log.info("click", x=x, y=y, screen_x=screen_x, screen_y=screen_y, intent=intent)

        elif action_type == "right_click":
            x, y = action_dict["x"], action_dict["y"]
            screen_x, screen_y = translate(x, y)
            pyautogui.rightClick(screen_x, screen_y)
            log.info("right_click", x=x, y=y, screen_x=screen_x, screen_y=screen_y, intent=intent)

        elif action_type == "press":
            key = action_dict["key"]
            pyautogui.press(key)
            log.info("press", key=key, intent=intent)

        elif action_type == "drag":
            x1, y1 = action_dict["x1"], action_dict["y1"]
            x2, y2 = action_dict["x2"], action_dict["y2"]
            sx1, sy1 = translate(x1, y1)
            sx2, sy2 = translate(x2, y2)
            pyautogui.moveTo(sx1, sy1)
            pyautogui.drag(sx2 - sx1, sy2 - sy1, duration=0.2)
            log.info("drag", x1=x1, y1=y1, x2=x2, y2=y2, intent=intent)

        elif action_type == "wait":
            ms = action_dict.get("ms", 100)
            await asyncio.sleep(ms / 1000)
            log.info("wait", ms=ms, intent=intent)

        else:
            log.warning("unknown_action", action_type=action_type, action=action_dict)
            return False

        # Small delay between actions for stability (async)
        await asyncio.sleep(config.action_delay)
        return True

    except KeyError as e:
        log.error("missing_action_param", action=action_dict, missing=str(e))
        return False
    except Exception as e:
        log.error("action_failed", action=action_dict, error=str(e))
        return False


async def execute_actions(actions: list[dict[str, Any] | Action]) -> int:
    """
    Execute a list of actions.

    Args:
        actions: List of action dictionaries or Action models

    Returns:
        Number of successfully executed actions
    """
    global _window_offset

    # Get window position before executing actions
    rect = get_game_window_rect()
    if rect:
        _window_offset = (rect[0], rect[1])
        log.debug("window_offset_updated", left=rect[0], top=rect[1])
    else:
        _window_offset = (0, 0)
        log.warning("window_rect_not_found", message="Using (0,0) offset")

    # Ensure game is focused before executing
    if not ensure_game_focused():
        log.warning("could_not_focus_before_actions")
        await asyncio.sleep(0.5)
        ensure_game_focused()  # Try once more

    success_count = 0
    for action in actions:
        if await execute_action(action):
            success_count += 1
    return success_count
