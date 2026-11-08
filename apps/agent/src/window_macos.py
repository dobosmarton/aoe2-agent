"""The native Mac game's window, read from the Quartz window list.

The Quartz list is used instead of NSWorkspace's front app, which goes stale in a
process without a run loop. On a MacBook with a camera notch, the game can leave a
black band under the menu bar. `game_area_points` cuts that band off, so frames
hold only the game.
"""

# pyobjc fills its framework modules at runtime, so a static checker can't see
# Quartz's and AppKit's functions and constants.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, cast

import AppKit
import Quartz
import structlog

from .window import GameGeometry, Platform

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

log = structlog.stdlib.get_logger()

GAME_NAME = "age of empires"
FOCUS_WAIT_SECONDS = 1.5
OPEN_WAIT_SECONDS = 2.0
_POLL_SECONDS = 0.1


def find_game_app() -> AppKit.NSRunningApplication | None:
    for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
        if GAME_NAME in (app.localizedName() or "").lower():
            return app
    return None


def _bounds(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    bounds = cast("Mapping[str, float]", row["kCGWindowBounds"])
    return (
        float(bounds["X"]),
        float(bounds["Y"]),
        float(bounds["Width"]),
        float(bounds["Height"]),
    )


def pick_game_window(rows: Iterable[Mapping[str, object]], pid: int) -> Mapping[str, object] | None:
    """The largest normal window the game owns. Menus and overlays sit on other layers."""
    own = [
        row
        for row in rows
        if row.get("kCGWindowOwnerPID") == pid and row.get("kCGWindowLayer") == 0
    ]

    def area(row: Mapping[str, object]) -> float:
        _, _, width, height = _bounds(row)
        return width * height

    return max(own, key=area, default=None)


def front_window_pid(rows: Iterable[Mapping[str, object]]) -> int | None:
    """The owner of the front normal window. The list is ordered front to back."""
    front = next((row for row in rows if row.get("kCGWindowLayer") == 0), None)
    if front is None:
        return None
    pid = front.get("kCGWindowOwnerPID")
    return int(pid) if isinstance(pid, int) else None


def game_area_points(
    bounds: tuple[float, float, float, float], menu_bar: float
) -> tuple[float, float, float, float]:
    """The window minus any strip under the menu bar, where the game draws only black."""
    x, y, width, height = bounds
    cut = min(max(menu_bar - y, 0.0), height)
    return (x, y + cut, width, height - cut)


def geometry_from(
    bounds: tuple[float, float, float, float], menu_bar: float, scale: float, window_id: int
) -> GameGeometry:
    x, y, width, height = game_area_points(bounds, menu_bar)
    return GameGeometry(
        round(x * scale),
        round(y * scale),
        round(width * scale),
        round(height * scale),
        scale,
        window_id,
    )


def wholly_on_screen(bounds: tuple[float, float, float, float], span: tuple[float, float]) -> bool:
    """Whether the window lies within the screens' horizontal span, in points.

    A full-screen game slides in from its own Space after focusing, and reports
    itself in front before the slide ends. Bounds read mid-slide put it partly off
    the screens, so input sent then would land in the wrong place.
    """
    x, _, width, _ = bounds
    return x >= span[0] and x + width <= span[1]


def screen_span_points() -> tuple[float, float]:
    """The leftmost and rightmost edges of all screens, in points."""
    frames = [screen.frame() for screen in AppKit.NSScreen.screens()]
    return (
        min(float(frame.origin.x) for frame in frames),
        max(float(frame.origin.x + frame.size.width) for frame in frames),
    )


def display_scale() -> float:
    """Pixels per point on the main display: 2 on a Retina panel."""
    mode = Quartz.CGDisplayCopyDisplayMode(Quartz.CGMainDisplayID())
    return Quartz.CGDisplayModeGetPixelWidth(mode) / Quartz.CGDisplayModeGetWidth(mode)


def menu_bar_points() -> float:
    """The menu bar's height in points.

    A full-screen app can hide the menu bar, which removes it from the visible frame.
    In that case the notch's safe-area inset gives the same strip.
    """
    screen = AppKit.NSScreen.mainScreen()
    frame, visible = screen.frame(), screen.visibleFrame()
    measured = float(frame.size.height - (visible.origin.y + visible.size.height))
    return measured if measured > 0 else float(screen.safeAreaInsets().top)


def screen_recording_allowed() -> bool:
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def input_allowed() -> bool:
    return bool(Quartz.CGPreflightPostEventAccess())


def keyboard_layouts() -> list[str]:
    """The selected keyboard layouts, for the probe's report."""
    sources = subprocess.run(
        ["defaults", "read", "com.apple.HIToolbox", "AppleSelectedInputSources"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [
        line.split("=")[1].strip(' ";')
        for line in sources.splitlines()
        if "KeyboardLayout Name" in line
    ]


def colour_space() -> str:
    return str(AppKit.NSScreen.mainScreen().colorSpace().localizedName())


def _window_rows(options: int) -> list[Mapping[str, object]]:
    rows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    return list(rows or [])


class MacBackend:
    """The game as a native macOS app. Input moves in points, frames are pixels."""

    @property
    def platform(self) -> Platform:
        return "macos"

    def geometry(self) -> GameGeometry | None:
        app = find_game_app()
        if app is None:
            return None
        row = pick_game_window(_window_rows(Quartz.kCGWindowListOptionAll), app.processIdentifier())
        if row is None:
            return None
        window_id = row.get("kCGWindowNumber")
        if not isinstance(window_id, int):
            return None
        return geometry_from(_bounds(row), menu_bar_points(), display_scale(), window_id)

    def input_scale(self) -> float:
        return display_scale()

    def in_front(self) -> bool:
        """The game leads the window list and has finished sliding onto the screen."""
        app = find_game_app()
        if app is None:
            return False
        pid = app.processIdentifier()
        rows = _window_rows(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        )
        if front_window_pid(rows) != pid:
            return False
        row = pick_game_window(rows, pid)
        return row is not None and wholly_on_screen(_bounds(row), screen_span_points())

    def is_running(self) -> bool:
        return self.geometry() is not None

    def focus(self, retries: int) -> bool:
        """Bring the game forward. Only startup calls this; input checks `in_front` instead."""
        app = find_game_app()
        if app is None:
            log.warning("game_window_not_found")
            return False
        for attempt in range(max(retries, 1)):
            if self.in_front():
                return True
            app.activateWithOptions_(AppKit.NSApplicationActivateAllWindows)
            if self._wait_for_front(FOCUS_WAIT_SECONDS):
                return True
            log.warning("focus_window_error", attempt=attempt + 1)
        # Activation can be refused while another app holds focus; `open` asks LaunchServices.
        subprocess.run(["open", "-b", str(app.bundleIdentifier())], check=False)
        return self._wait_for_front(OPEN_WAIT_SECONDS)

    def _wait_for_front(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.in_front():
                return True
            time.sleep(_POLL_SECONDS)
        return self.in_front()
