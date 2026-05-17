"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import pyautogui
import structlog
from pydantic import BaseModel

from .config import config
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.get_logger()

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
    log.debug("detected_entities_set", count=len(_detected_entities))


def get_detected_entities() -> list[dict]:
    """Return the current detected entity list."""
    return _detected_entities


def clear_detected_entities() -> None:
    """Clear the cached detected entities."""
    global _detected_entities
    _detected_entities = []


def _build_executor_rng() -> random.Random:
    """Fresh local RNG seeded from config.seed.

    Read at call time so tests that mutate config.seed pick up the change
    without re-importing. Never seeds the global `random` module — keeps the
    determinism contract local to the build-retry helper.
    """
    return random.Random(config.seed)


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
    """Resolve action coordinates from target_id, target_class, or x/y fields.

    Returns (error_detail, coords). error_detail is non-empty on failure.
    """
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


def _translate(x: int, y: int) -> tuple[int, int]:
    """Translate screenshot-relative coords to screen-absolute."""
    return (x + _window_offset[0], y + _window_offset[1])


# ---------------------------------------------------------------------------
# Per-type action handlers
# ---------------------------------------------------------------------------

BUILD_PLACEMENT_KEYWORDS = ("place", "build")
# Random angular retries — large enough to escape dense tree/building clusters
# that broke the old 80px cardinal-offset retry (exp_0013 logged 63 retries
# across 35 turns, many on the same blocked terrain).
BUILD_RETRY_RADIUS_MIN = 250
BUILD_RETRY_RADIUS_MAX = 350
BUILD_RETRY_ATTEMPTS = 2
BUILD_SETTLE_DELAY = 0.15
BUILD_RETRY_DELAY = 0.1
RESCAN_SETTLE_DELAY = 0.3
DEFAULT_WAIT_MS = 100

# Module-level cumulative retry telemetry (resets per process / per game).
# Surfaced via build_placement_retry log lines so the user can grep
# `total_count`/`total_seconds` to see how much turn budget got eaten by
# failed placements.
_build_retry_total_seconds: float = 0.0
_build_retry_count: int = 0


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

    # Building placement retry — if first click was invalid (tree/building/water),
    # try `BUILD_RETRY_ATTEMPTS` random angular offsets at 250-350 px from the
    # original. Replaces the previous 4 cardinal 80 px offsets, which often hit
    # the same blocked terrain because tree clusters and building footprints
    # are larger than 80 px.
    intent_lower = intent.lower()
    if any(word in intent_lower for word in BUILD_PLACEMENT_KEYWORDS):
        global _build_retry_total_seconds, _build_retry_count
        retry_start = time.monotonic()
        await asyncio.sleep(BUILD_SETTLE_DELAY)
        offsets: list[tuple[int, int]] = []
        rng = _build_executor_rng()
        for _ in range(BUILD_RETRY_ATTEMPTS):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(BUILD_RETRY_RADIUS_MIN, BUILD_RETRY_RADIUS_MAX)
            dx = int(radius * math.cos(angle))
            dy = int(radius * math.sin(angle))
            offsets.append((dx, dy))
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

    return ActionResult(True, "ok")


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


# Dispatch table: action type -> handler
_ACTION_HANDLERS: dict[
    str,
    Callable[[dict[str, object], str], Awaitable[ActionResult]],
] = {
    "click": _handle_click,
    "right_click": _handle_right_click,
    "press": _handle_press,
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
        action_dict = action.model_dump()  # pyright: ignore[reportAttributeAccessIssue]
    else:
        validated = validate_action(action)
        if not validated:
            log.warning("invalid_action", action=action)
            return ActionResult(False, "invalid action format")
        action_dict = validated.model_dump()

    action_type = action_dict.get("type", "")
    intent = action_dict.get("intent", "")

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


async def execute_actions(actions: Sequence[dict[str, object] | Action]) -> list[ActionResult]:
    """Execute a list of actions sequentially."""
    if not ensure_game_focused():
        log.warning("could_not_focus_before_actions")
        await asyncio.sleep(0.5)
        ensure_game_focused()

    return [await execute_action(action) for action in actions]
