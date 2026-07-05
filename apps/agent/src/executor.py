"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

import pyautogui
import structlog
from pydantic import BaseModel

from .config import config
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.stdlib.get_logger()

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
    log.debug("detected_entities_set", count=len(_detected_entities))


def get_detected_entities() -> list[dict]:
    """Return the current detected entity list."""
    return _detected_entities


def clear_detected_entities() -> None:
    """Clear the cached detected entities."""
    global _detected_entities
    _detected_entities = []


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
BUILD_RING_DIRECTIONS: int = 8
BUILD_CLUTTER_RADIUS: int = 160  # entities within this of a candidate = clutter

# Play-area margins (screenshot px). The HUD occupies the top resource bar and the
# bottom command panel; those regions have no entities, so an emptiness score would
# wrongly rank them as "open" — exclude them from build candidates and gather clicks.
UI_MARGIN_TOP: int = 160
UI_MARGIN_BOTTOM: int = 240
UI_MARGIN_SIDE: int = 40

# Economic build-menu key → detected building class, used to verify a placement
# actually landed (the class appears in the entity cache after a rescan).
BUILD_KEY_TO_CLASS: dict[str, str] = {
    "q": "house",
    "w": "mill",
    "e": "mining_camp",
    "r": "lumber_camp",
    "a": "farm",
    "s": "blacksmith",
    "t": "dock",
}

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

    Lower = emptier ground = more likely a valid building spot.
    """
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(
        1
        for entity in _detected_entities
        if (center := entity.get("center")) and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
    )


def _open_ground_candidates(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """Ring points around `anchor` that lie in the play area, emptiest-first."""
    ax, ay = anchor
    min_x, min_y, max_x, max_y = _play_area_bounds()
    candidates: list[tuple[int, int]] = []
    for radius in BUILD_RING_RADII:
        for i in range(BUILD_RING_DIRECTIONS):
            angle = 2.0 * math.pi * i / BUILD_RING_DIRECTIONS
            cx = int(ax + radius * math.cos(angle))
            cy = int(ay + radius * math.sin(angle))
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                candidates.append((cx, cy))
    candidates.sort(key=_clutter_score)
    return candidates


def default_build_placement() -> tuple[int, int]:
    """Screenshot-relative point to start a building placement when the model
    gave no x,y — the executor picks *where*, since the text-only model can't see
    open ground.

    Anchors on the detected town centre but returns the emptiest ring point *around*
    it, never the TC tile itself (clicking on the TC always fails). Falls back to the
    window centre's open ring, then a fixed point, when no TC is detected.
    """
    anchor = _resolve_target_class("town_center")
    if anchor is None:
        width, height = _window_size()
        anchor = (width // 2, height // 2)
    candidates = _open_ground_candidates(anchor)
    return candidates[0] if candidates else anchor


def build_steps(
    building_key: str, intent: str, placement: tuple[int, int]
) -> list[dict[str, object]]:
    """Press/click sequence for a build: select idle villager → open the economic
    build menu → pick the building → place it.

    Shared by the single-shot build handler (`_handle_build`) and the tool-loop
    build composite (`claude.ClaudeProvider._execute_build`) so the steps live in
    exactly one place.
    """
    place_x, place_y = placement
    return [
        {"type": "press", "key": ".", "intent": f"Select idle villager ({intent})"},
        {"type": "press", "key": "q", "intent": "Open economic build menu"},
        {"type": "press", "key": building_key, "intent": f"Select building ({intent})"},
        {
            "type": "click",
            "x": place_x,
            "y": place_y,
            "building_key": building_key,  # lets _handle_click verify the placement landed
            "intent": f"Place building ({intent})",
        },
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

    # Building placement: the anchor is already chosen on open ground
    # (default_build_placement), so spray a few deterministic compass offsets to
    # escape small local blockage, then right-click to cancel the leftover ghost.
    # A build placement consumes at the first valid tile, so extra clicks are inert
    # ground clicks. If the build carries a building_key, verify it actually landed.
    intent_lower = intent.lower()
    if any(word in intent_lower for word in BUILD_PLACEMENT_KEYWORDS):
        global _build_retry_total_seconds, _build_retry_count
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
        if isinstance(building_key, str):
            await _verify_build_placement(building_key, (x, y))

    return ActionResult(True, "ok")


def _compass_offsets(radius: int, count: int) -> list[tuple[int, int]]:
    """`count` evenly-spaced (dx, dy) offsets at `radius` px — a deterministic spray."""
    return [
        (int(radius * math.cos(a)), int(radius * math.sin(a)))
        for i in range(count)
        for a in (2.0 * math.pi * i / count,)
    ]


async def _verify_build_placement(building_key: str, point: tuple[int, int]) -> None:
    """Rescan and log whether a building of the expected class landed near `point`.

    Best-effort (needs a rescan callback). Turns the old silent placement failure
    into a greppable build_placement_verified / build_placement_failed signal so the
    next turn can react instead of re-issuing the same doomed build.
    """
    expected = BUILD_KEY_TO_CLASS.get(building_key)
    if expected is None or _rescan_fn is None:
        return
    await asyncio.sleep(RESCAN_SETTLE_DELAY)
    await _rescan_fn()
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    landed = any(
        entity.get("class") == expected
        and (center := entity.get("center"))
        and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
        for entity in _detected_entities
    )
    if landed:
        log.info("build_placement_verified", building=expected, x=px, y=py)
    else:
        log.warning("build_placement_failed", building=expected, x=px, y=py)


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
    for step in build_steps(key, intent, default_build_placement()):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"build failed at: {step.get('intent', '')}")
    return ActionResult(True, f"built ({intent})")


# Dispatch table: action type -> handler
_ACTION_HANDLERS: dict[
    str,
    Callable[[dict[str, object], str], Awaitable[ActionResult]],
] = {
    "click": _handle_click,
    "right_click": _handle_right_click,
    "press": _handle_press,
    "build": _handle_build,
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


async def execute_actions(actions: Sequence[dict[str, object] | Action]) -> list[ActionResult]:
    """Execute a list of actions sequentially."""
    if not ensure_game_focused():
        log.warning("could_not_focus_before_actions")
        await asyncio.sleep(0.5)
        ensure_game_focused()

    return [await execute_action(action) for action in actions]
