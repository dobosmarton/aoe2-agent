"""Window management for AoE2 LLM Agent."""

import time

import structlog

log = structlog.get_logger()

# Try to import pygetwindow, fall back gracefully if not available
try:
    import pygetwindow as gw

    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    log.warning("pygetwindow_not_available", message="Window management disabled")

AOE2_WINDOW_TITLE = "Age of Empires II: Definitive Edition"


def find_game_window():
    """
    Find the AoE2 window.

    Returns the window object or None if not found.
    """
    if not PYGETWINDOW_AVAILABLE:
        return None

    try:
        windows = gw.getWindowsWithTitle(AOE2_WINDOW_TITLE)
        return windows[0] if windows else None
    except Exception as e:
        log.warning("find_window_error", error=str(e))
        return None


def ensure_game_focused(retries: int = 3) -> bool:
    """
    Ensure AoE2 window is focused.

    Args:
        retries: Number of retry attempts if focus fails

    Returns True if game is focused, False otherwise.
    """
    if not PYGETWINDOW_AVAILABLE:
        # If pygetwindow isn't available, assume game is focused
        return True

    window = find_game_window()
    if not window:
        log.warning("game_window_not_found")
        return False

    for attempt in range(retries):
        try:
            if window.isActive:
                return True

            window.activate()
            time.sleep(0.2)  # Wait for focus to take effect

            if window.isActive:
                return True

        except Exception as e:
            log.warning("focus_window_error", error=str(e), attempt=attempt + 1)

        if attempt < retries - 1:
            time.sleep(0.3)  # Wait before retry

    # Final check
    try:
        return window.isActive
    except Exception:
        return False


def is_game_running() -> bool:
    """
    Check if the game window exists.

    Returns True if game window is found.
    """
    if not PYGETWINDOW_AVAILABLE:
        # If pygetwindow isn't available, assume game is running
        return True

    return find_game_window() is not None


def get_game_window_rect() -> tuple[int, int, int, int] | None:
    """
    Get the game window rectangle.

    Returns (left, top, width, height) or None if not found.
    """
    if not PYGETWINDOW_AVAILABLE:
        return None

    window = find_game_window()
    if not window:
        return None

    try:
        return (window.left, window.top, window.width, window.height)
    except Exception as e:
        log.warning("get_window_rect_error", error=str(e))
        return None
