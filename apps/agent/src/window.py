"""Find, focus and measure the game window on Windows and macOS.

Rectangles here are in frame pixels, the pixels a capture returns. The mouse can
move in other units: on a Retina Mac it moves in points, two pixels each. So a
frame position goes through `to_input_point` before it reaches pyautogui.

The platform backend is picked on first use. Tests replace it with
`NullBackend`, so they never see a real game window.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

log = structlog.stdlib.get_logger()

# Try to import pygetwindow, fall back gracefully if not available
try:
    import pygetwindow as gw

    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    log.warning("pygetwindow_not_available", message="Window management disabled")

AOE2_WINDOW_TITLE = "Age of Empires II: Definitive Edition"
# Pause between minimize and restore in the foreground-lock workaround.
_MINIMIZE_RESTORE_DELAY_S = 0.1

Platform = Literal["windows", "macos", "none"]


def to_input_point(x: float, y: float, left: float, top: float, scale: float) -> tuple[int, int]:
    """A frame position as an input position: add the window origin, then divide by the scale."""
    return (round((x + left) / scale), round((y + top) / scale))


@dataclass(frozen=True, slots=True)
class GameGeometry:
    """Where the game's frame sits on screen, in frame pixels.

    `scale` is frame pixels per input unit: 1 on Windows, 2 on a Retina Mac.
    `window_id` is the Quartz window number on macOS, which capture needs.
    """

    left: int
    top: int
    width: int
    height: int
    scale: float = 1.0
    window_id: int | None = None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def to_input(self, x: float, y: float) -> tuple[int, int]:
        return to_input_point(x, y, self.left, self.top, self.scale)

    def to_frame(self, x: float, y: float) -> tuple[int, int]:
        return (round(x * self.scale) - self.left, round(y * self.scale) - self.top)


class WindowBackend(Protocol):
    @property
    def platform(self) -> Platform: ...
    def geometry(self) -> GameGeometry | None: ...
    def input_scale(self) -> float: ...
    def focus(self, retries: int) -> bool: ...
    def in_front(self) -> bool: ...
    def is_running(self) -> bool: ...


class GameWindow(Protocol):
    """Subset of the pygetwindow Window API that this module relies on.

    Defined as a Protocol because pygetwindow's Window class differs by
    platform (Win32Window on Windows, nothing on Linux) and the import is
    conditional. Structural typing avoids the platform-specific class.
    """

    @property
    def isActive(self) -> bool: ...
    @property
    def left(self) -> int: ...
    @property
    def top(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def activate(self) -> None: ...
    def minimize(self) -> None: ...
    def restore(self) -> None: ...


def find_game_window() -> GameWindow | None:
    """
    Find the AoE2 window on Windows.

    Returns the window object or None if not found.
    """
    if not PYGETWINDOW_AVAILABLE:
        return None

    try:
        # getWindowsWithTitle is the Windows backend's API; pygetwindow stubs
        # only expose cross-platform attrs.
        windows = gw.getWindowsWithTitle(AOE2_WINDOW_TITLE)  # pyright: ignore[reportAttributeAccessIssue]
        return windows[0] if windows else None
    except Exception as e:
        log.warning("find_window_error", error=str(e))
        return None


class WindowsBackend:
    """The game in a Windows session, through pygetwindow. Input and frames share pixels."""

    @property
    def platform(self) -> Platform:
        return "windows"

    def geometry(self) -> GameGeometry | None:
        window = find_game_window()
        if not window:
            return None

        try:
            return GameGeometry(window.left, window.top, window.width, window.height)
        except Exception as e:
            log.warning("get_window_rect_error", error=str(e))
            return None

    def input_scale(self) -> float:
        return 1.0

    def focus(self, retries: int) -> bool:
        window = find_game_window()
        if not window:
            log.warning("game_window_not_found")
            return False

        for attempt in range(retries):
            try:
                if window.isActive:
                    return True

                if attempt == 0:
                    window.activate()
                else:
                    # activate() was refused (Windows foreground lock — the log
                    # shows "Error code from Windows: 0 ... completed successfully"
                    # while nothing happens). A minimize/restore cycle forces the
                    # shell to re-foreground the window.
                    window.minimize()
                    time.sleep(_MINIMIZE_RESTORE_DELAY_S)
                    window.restore()
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

    def in_front(self) -> bool:
        window = find_game_window()
        try:
            return bool(window and window.isActive)
        except Exception:
            return False

    def is_running(self) -> bool:
        return find_game_window() is not None


class NullBackend:
    """No window API: assume the game runs and has focus, with no known rectangle.

    This was the behaviour without pygetwindow, and tests use it so a game left open
    on the machine can't change their results.
    """

    @property
    def platform(self) -> Platform:
        return "none"

    def geometry(self) -> GameGeometry | None:
        return None

    def input_scale(self) -> float:
        return 1.0

    def focus(self, retries: int) -> bool:
        del retries
        return True

    def in_front(self) -> bool:
        return True

    def is_running(self) -> bool:
        return True


def _default_backend() -> WindowBackend:
    if sys.platform == "darwin":
        try:
            from .window_macos import MacBackend
        except ImportError as error:
            log.warning("macos_window_api_unavailable", error=str(error))
            return NullBackend()
        return MacBackend()
    if PYGETWINDOW_AVAILABLE:
        return WindowsBackend()
    return NullBackend()


_backend: WindowBackend | None = None


def _current() -> WindowBackend:
    global _backend
    if _backend is None:
        _backend = _default_backend()
    return _backend


@contextmanager
def use_backend(backend: WindowBackend) -> Iterator[None]:
    """Swap the window backend for a block, e.g. a scripted scenario with no game."""
    global _backend
    previous = _backend
    _backend = backend
    try:
        yield
    finally:
        _backend = previous


def game_platform() -> Platform:
    return _current().platform


def get_game_geometry() -> GameGeometry | None:
    return _current().geometry()


def get_game_window_rect() -> tuple[int, int, int, int] | None:
    """
    Get the game window rectangle in frame pixels.

    Returns (left, top, width, height) or None if not found.
    """
    geometry = get_game_geometry()
    return geometry.rect if geometry is not None else None


def get_input_scale() -> float:
    """Frame pixels per input unit: 1 on Windows, 2 on a Retina Mac."""
    return _current().input_scale()


def ensure_game_focused(retries: int = 3) -> bool:
    """
    Ensure AoE2 window is focused.

    Args:
        retries: Number of retry attempts if focus fails

    Returns True if game is focused, False otherwise.
    """
    return _current().focus(retries)


def game_in_front() -> bool:
    """Whether the game is the front window right now. Never changes focus."""
    return _current().in_front()


def is_game_running() -> bool:
    """
    Check if the game window exists.

    Returns True if game window is found.
    """
    return _current().is_running()
