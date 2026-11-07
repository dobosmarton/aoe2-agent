#!/usr/bin/env python
"""Probe the native Mac game before the agent depends on it.

Run from Terminal.app, which holds the Screen Recording and Accessibility permissions,
with the game running full screen:

    uv run python scripts/macos_probe.py info
    uv run python scripts/macos_probe.py capture
    uv run python scripts/macos_probe.py snap tc_selected
    uv run python scripts/macos_probe.py input   # moves the mouse and presses keys in the game

Everything goes to logs/mac-probe/: a report.json that each command adds to, and
lossless PNGs. Only `input` sends anything to the game.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import AppKit
import CoreMedia
import mss
import mss.darwin
import numpy as np
import objc
import Quartz
import ScreenCaptureKit
from Foundation import NSObject
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from mss.base import MSSBase

GAME_NAME = "age of empires"
OUTPUT = Path("logs/mac-probe")
REPORT = OUTPUT / "report.json"
GRABS = 50
BAND_ROWS = 10
BAND_BRIGHTNESS = 8.0  # Mean value below this means the top rows are black.
CLICK_HOLD_SECONDS = 0.04
SCROLL_GAP_SECONDS = 0.03
ZOOM_OUT_STEPS = 30
ZOOM_IN_STEPS = 12
SETTLE_SECONDS = 0.5
FOCUS_WAIT_SECONDS = 1.5
REST_WAIT_SECONDS = 3.0  # A full-screen game slides in from its own Space after focusing.
REST_POLL_SECONDS = 0.15
WINDOW_OPTIONS = (
    Quartz.kCGWindowImageBoundsIgnoreFraming
    | Quartz.kCGWindowImageShouldBeOpaque
    | Quartz.kCGWindowImageBestResolution
)
MSS_OPTIONS = Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageShouldBeOpaque
SCK_FPS = 30
SCK_QUEUE_DEPTH = 4
SCK_ANSWER_SECONDS = 5.0
SCK_GRAB_GAP_SECONDS = 0.1  # About the agent's pace, so a grab usually finds a new frame.


def game_app() -> AppKit.NSRunningApplication | None:
    for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
        if GAME_NAME in (app.localizedName() or "").lower():
            return app
    return None


def pick_game_window(rows: list[dict[str, Any]], pid: int) -> dict[str, Any] | None:
    """The largest normal window the game owns. Menus and overlays sit on other layers."""
    own = [
        row
        for row in rows
        if row.get("kCGWindowOwnerPID") == pid and row.get("kCGWindowLayer") == 0
    ]

    def area(row: dict[str, Any]) -> float:
        bounds = row["kCGWindowBounds"]
        return float(bounds["Width"]) * float(bounds["Height"])

    return max(own, key=area, default=None)


def game_window(pid: int) -> dict[str, Any] | None:
    rows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    return pick_game_window(list(rows), pid)


def frontmost_is_game(pid: int) -> bool:
    """The front window, read from the window list: the app API goes stale without a run loop."""
    rows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    front = next((row for row in rows if row.get("kCGWindowLayer") == 0), None)
    return front is not None and front.get("kCGWindowOwnerPID") == pid


def bring_to_front(app: AppKit.NSRunningApplication) -> bool:
    pid = app.processIdentifier()
    app.activateWithOptions_(AppKit.NSApplicationActivateAllWindows)
    deadline = time.monotonic() + FOCUS_WAIT_SECONDS
    while time.monotonic() < deadline:
        if frontmost_is_game(pid):
            return True
        time.sleep(0.1)
    subprocess.run(["open", "-b", app.bundleIdentifier()], check=False)
    time.sleep(2.0)
    return frontmost_is_game(pid)


def at_rest(
    previous: dict[str, float] | None, current: dict[str, float] | None, screen_width: float
) -> bool:
    """Still in the same place as last time, and wholly on the screen.

    The game reports itself in front before its slide-in ends, and bounds read mid-slide
    put it half off the screen.
    """
    if current is None or current != previous:
        return False
    return current["X"] >= 0 and current["X"] + current["Width"] <= screen_width


def resting_window(app: AppKit.NSRunningApplication) -> dict[str, Any]:
    """The game window once it has stopped moving. Stops the probe if it never does."""
    pid = app.processIdentifier()
    screen_width = float(AppKit.NSScreen.mainScreen().frame().size.width)
    previous: dict[str, float] | None = None
    deadline = time.monotonic() + REST_WAIT_SECONDS
    while time.monotonic() < deadline:
        window = game_window(pid)
        current = dict(window["kCGWindowBounds"]) if window else None
        if window is not None and at_rest(previous, current, screen_width):
            return window
        previous = current
        time.sleep(REST_POLL_SECONDS)
    raise SystemExit(
        "The game window kept moving or stayed off the screen, so nothing was sent or measured."
    )


def display() -> dict[str, Any]:
    main = Quartz.CGMainDisplayID()
    mode = Quartz.CGDisplayCopyDisplayMode(main)
    screen = AppKit.NSScreen.mainScreen()
    frame, visible = screen.frame(), screen.visibleFrame()
    sources = subprocess.run(
        ["defaults", "read", "com.apple.HIToolbox", "AppleSelectedInputSources"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    layouts = [
        line.split("=")[1].strip(' ";')
        for line in sources.splitlines()
        if "KeyboardLayout Name" in line
    ]
    return {
        "points": [Quartz.CGDisplayModeGetWidth(mode), Quartz.CGDisplayModeGetHeight(mode)],
        "pixels": [
            Quartz.CGDisplayModeGetPixelWidth(mode),
            Quartz.CGDisplayModeGetPixelHeight(mode),
        ],
        "scale": Quartz.CGDisplayModeGetPixelWidth(mode) / Quartz.CGDisplayModeGetWidth(mode),
        "menu_bar_points": frame.size.height - (visible.origin.y + visible.size.height),
        "safe_area_top_points": screen.safeAreaInsets().top,
        "colour_space": screen.colorSpace().localizedName(),
        "keyboard_layouts": layouts,
        "screen_recording_allowed": bool(Quartz.CGPreflightScreenCaptureAccess()),
        "input_allowed": bool(Quartz.CGPreflightPostEventAccess()),
    }


def bgra_to_rgb(raw: bytes, width: int, height: int, row_bytes: int) -> np.ndarray:
    # Some buffers carry spare bytes after the last row.
    pixels = np.frombuffer(raw, dtype=np.uint8, count=row_bytes * height)
    pixels = pixels.reshape(height, row_bytes // 4, 4)[:, :width]
    return np.ascontiguousarray(pixels[..., 2::-1])


def grab_window(window_id: int) -> np.ndarray:
    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, window_id, WINDOW_OPTIONS
    )
    if image is None:
        raise RuntimeError("window capture returned nothing")
    return cgimage_to_rgb(image)


def cgimage_to_rgb(image: object) -> np.ndarray:
    raw = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image)))
    return bgra_to_rgb(
        raw,
        Quartz.CGImageGetWidth(image),
        Quartz.CGImageGetHeight(image),
        Quartz.CGImageGetBytesPerRow(image),
    )


def _answer(ask: Callable[[Callable[..., None]], None]) -> tuple[object, ...]:
    """Call a ScreenCaptureKit method that answers through a completion handler."""
    answered = threading.Event()
    answers: list[tuple[object, ...]] = []

    def handler(*values: object) -> None:
        answers.append(values)
        answered.set()

    ask(handler)
    if not answered.wait(SCK_ANSWER_SECONDS):
        raise RuntimeError("ScreenCaptureKit didn't answer")
    return answers[0]


def sck_source(window_id: int, width: int, height: int) -> tuple[object, object]:
    """A filter for the game window alone, and a full-resolution BGRA configuration."""
    content, error = _answer(
        lambda handler: (
            ScreenCaptureKit.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                True, True, handler
            )
        )
    )
    if error is not None:
        raise RuntimeError(f"ScreenCaptureKit refused: {error}")
    window = next((item for item in content.windows() if item.windowID() == window_id), None)
    if window is None:
        raise RuntimeError("ScreenCaptureKit can't see the game window")
    source = ScreenCaptureKit.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
    config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    config.setWidth_(width)
    config.setHeight_(height)
    config.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
    config.setShowsCursor_(False)
    config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, SCK_FPS))
    config.setQueueDepth_(SCK_QUEUE_DEPTH)
    return source, config


def sck_screenshot(source: object, config: object) -> np.ndarray:
    image, error = _answer(
        lambda handler: (
            ScreenCaptureKit.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
                source, config, handler
            )
        )
    )
    if image is None:
        raise RuntimeError(f"ScreenCaptureKit screenshot failed: {error}")
    return cgimage_to_rgb(image)


def pixel_buffer_to_rgb(buffer: object) -> np.ndarray:
    Quartz.CVPixelBufferLockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        row_bytes = Quartz.CVPixelBufferGetBytesPerRow(buffer)
        height = Quartz.CVPixelBufferGetHeight(buffer)
        width = Quartz.CVPixelBufferGetWidth(buffer)
        base = Quartz.CVPixelBufferGetBaseAddress(buffer)
        raw = bytes(base.as_buffer(row_bytes * height))
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    return bgra_to_rgb(raw, width, height, row_bytes)


class LatestFrame(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
    """Keeps the newest frame the stream delivered. Unchanged screens deliver none."""

    def init(self) -> LatestFrame | None:
        self = objc.super(LatestFrame, self).init()
        if self is None:
            return None
        self.lock = threading.Lock()
        self.buffer = None
        self.delivered = 0
        return self

    def stream_didOutputSampleBuffer_ofType_(
        self, _stream: object, sample: object, kind: int
    ) -> None:
        if kind != ScreenCaptureKit.SCStreamOutputTypeScreen:
            return
        buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        if buffer is None:
            return
        with self.lock:
            self.buffer = buffer
            self.delivered += 1

    def take(self) -> object:
        with self.lock:
            return self.buffer


def sck_stream_result(source: object, config: object) -> dict[str, Any]:
    """Stream the window, and time taking the newest frame at roughly the agent's pace."""
    try:
        stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
            source, config, None
        )
        latest = LatestFrame.alloc().init()
        added, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            latest, ScreenCaptureKit.SCStreamOutputTypeScreen, None, None
        )
        if not added:
            raise RuntimeError(f"ScreenCaptureKit wouldn't add the output: {error}")
        (error,) = _answer(stream.startCaptureWithCompletionHandler_)
        if error is not None:
            raise RuntimeError(f"ScreenCaptureKit wouldn't start: {error}")
    except Exception as error:  # Report the failure instead of hiding the other methods.
        return {"error": f"{type(error).__name__}: {error}"}
    try:
        deadline = time.monotonic() + SCK_ANSWER_SECONDS
        while latest.take() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if latest.take() is None:
            return {"error": "the stream delivered no frame"}
        started, first_count = time.monotonic(), latest.delivered
        result = _method_result(
            "sck_stream",
            lambda: pixel_buffer_to_rgb(latest.take()),
            pause=SCK_GRAB_GAP_SECONDS,
        )
        seconds = time.monotonic() - started
        result["frames_per_second"] = round((latest.delivered - first_count) / seconds, 1)
        return result
    finally:
        _answer(stream.stopCaptureWithCompletionHandler_)


def grab_region(sct: MSSBase, bounds: dict[str, float]) -> np.ndarray:
    mss.darwin.IMAGE_OPTIONS = MSS_OPTIONS  # Without the nominal-resolution flag: full pixels.
    region = {
        "left": int(bounds["X"]),
        "top": int(bounds["Y"]),
        "width": int(bounds["Width"]),
        "height": int(bounds["Height"]),
    }
    shot = sct.grab(region)
    return bgra_to_rgb(bytes(shot.bgra), shot.width, shot.height, shot.width * 4)


def top_is_black(frame: np.ndarray) -> bool:
    return float(frame[:BAND_ROWS].mean()) < BAND_BRIGHTNESS


def save(frame: np.ndarray, name: str) -> str:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.png"
    Image.fromarray(frame).save(path)
    return str(path)


def update_report(section: str, values: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[section] = values
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({section: values}, indent=2, default=str))


def require_game() -> tuple[AppKit.NSRunningApplication, dict[str, Any]]:
    app = game_app()
    if app is None:
        raise SystemExit("The game isn't running. Start a skirmish first.")
    window = game_window(app.processIdentifier())
    if window is None:
        raise SystemExit("The game is running but has no normal window yet.")
    return app, window


def window_summary(app: AppKit.NSRunningApplication, window: dict[str, Any]) -> dict[str, Any]:
    return {
        "app": app.localizedName(),
        "bundle_id": app.bundleIdentifier(),
        "pid": app.processIdentifier(),
        "window_id": window["kCGWindowNumber"],
        "bounds_points": dict(window["kCGWindowBounds"]),
        "on_screen": bool(window.get("kCGWindowIsOnscreen")),
    }


def info_command(_: argparse.Namespace) -> None:
    values: dict[str, Any] = {"display": display()}
    app = game_app()
    if app is not None:
        window = game_window(app.processIdentifier())
        values["game"] = (
            window_summary(app, window) if window else {"app": app.localizedName(), "window": None}
        )
    update_report("info", values)


def _timed(grab: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    frame = grab()
    return frame, (time.perf_counter() - started) * 1000


def _method_result(name: str, grab: Callable[[], np.ndarray], pause: float = 0.0) -> dict[str, Any]:
    """Time each grab alone. `pause` waits between grabs, outside the timing."""
    try:
        timings: list[float] = []
        frame = None
        for _ in range(GRABS):
            time.sleep(pause)
            frame, took = _timed(grab)
            timings.append(took)
    except Exception as error:  # Report the failure instead of hiding the other method.
        return {"error": f"{type(error).__name__}: {error}"}
    assert frame is not None
    timings.sort()
    return {
        "size": [int(frame.shape[1]), int(frame.shape[0])],
        "p50_ms": round(statistics.median(timings), 1),
        "p95_ms": round(timings[int(0.95 * (len(timings) - 1))], 1),
        "top_rows_black": top_is_black(frame),
        "mean_brightness": round(float(frame.mean()), 1),
        "png": save(frame, f"capture_{name}"),
    }


def capture_command(_: argparse.Namespace) -> None:
    app, window = require_game()
    front = bring_to_front(app)
    window = resting_window(app)
    with mss.MSS() as sct:
        results = {
            "game_in_front": front,
            "window": window_summary(app, window),
            "window_id_capture": _method_result(
                "window", lambda: grab_window(window["kCGWindowNumber"])
            ),
            "mss_region_capture": _method_result(
                "mss", lambda: grab_region(sct, window["kCGWindowBounds"])
            ),
        }
    scale = float(AppKit.NSScreen.mainScreen().backingScaleFactor())
    bounds = window["kCGWindowBounds"]
    try:
        source, config = sck_source(
            window["kCGWindowNumber"],
            round(bounds["Width"] * scale),
            round(bounds["Height"] * scale),
        )
    except Exception as error:  # Report the failure instead of hiding the other methods.
        results["sck_screenshot"] = results["sck_stream"] = {
            "error": f"{type(error).__name__}: {error}"
        }
    else:
        results["sck_screenshot"] = _method_result(
            "sck_screenshot", lambda: sck_screenshot(source, config)
        )
        results["sck_stream"] = sck_stream_result(source, config)
    update_report("capture", results)


def snap_command(arguments: argparse.Namespace) -> None:
    app, _ = require_game()
    bring_to_front(app)
    window = resting_window(app)
    time.sleep(SETTLE_SECONDS)
    frame = grab_window(window["kCGWindowNumber"])
    update_report(
        f"snap_{arguments.name}",
        {"size": list(frame.shape[1::-1]), "png": save(frame, f"snap_{arguments.name}")},
    )


def input_command(_: argparse.Namespace) -> None:
    import pyautogui

    if not Quartz.CGPreflightPostEventAccess():
        raise SystemExit(
            "This terminal can't send input yet: allow it in System Settings > Privacy & Security > Accessibility."
        )
    app = require_game()[0]
    pid = app.processIdentifier()
    if not bring_to_front(app):
        raise SystemExit("Couldn't bring the game to the front, so nothing was sent.")
    window = resting_window(app)
    bounds = window["kCGWindowBounds"]
    centre = (bounds["X"] + bounds["Width"] / 2, bounds["Y"] + bounds["Height"] / 2)
    window_id = window["kCGWindowNumber"]
    results: dict[str, Any] = {"window": window_summary(app, window)}

    def still_in_front() -> None:
        if not frontmost_is_game(pid):
            update_report("input", {**results, "stopped": "the game left the front"})
            raise SystemExit("The game is no longer in front, so the probe stopped sending input.")

    pyautogui.moveTo(*centre)
    time.sleep(0.2)
    position = pyautogui.position()
    results["cursor_target_points"] = list(centre)
    results["cursor_read_back_points"] = [position.x, position.y]

    still_in_front()
    pyautogui.press("h")
    time.sleep(SETTLE_SECONDS)
    results["after_h"] = save(grab_window(window_id), "input_after_h")

    still_in_front()
    pyautogui.press("escape")
    time.sleep(SETTLE_SECONDS)
    results["after_escape"] = save(grab_window(window_id), "input_after_escape")
    pyautogui.mouseDown(*centre)
    time.sleep(CLICK_HOLD_SECONDS)
    pyautogui.mouseUp(*centre)
    time.sleep(SETTLE_SECONDS)
    results["after_click"] = save(grab_window(window_id), "input_after_click")

    still_in_front()
    pyautogui.press("h")
    time.sleep(SETTLE_SECONDS)
    for _ in range(ZOOM_OUT_STEPS):
        pyautogui.scroll(-1)
        time.sleep(SCROLL_GAP_SECONDS)
    time.sleep(SETTLE_SECONDS)
    ladder = [save(grab_window(window_id), "zoom_00")]
    for step in range(1, ZOOM_IN_STEPS + 1):
        still_in_front()
        pyautogui.scroll(1)
        time.sleep(0.4)
        ladder.append(save(grab_window(window_id), f"zoom_{step:02d}"))
    results["zoom_ladder"] = ladder
    results["zoom_recipe"] = f"{ZOOM_OUT_STEPS} x scroll(-1), then scroll(+1) per step"
    update_report("input", results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info").set_defaults(run=info_command)
    commands.add_parser("capture").set_defaults(run=capture_command)
    snap = commands.add_parser("snap")
    snap.add_argument("name")
    snap.set_defaults(run=snap_command)
    commands.add_parser("input").set_defaults(run=input_command)
    arguments = parser.parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
