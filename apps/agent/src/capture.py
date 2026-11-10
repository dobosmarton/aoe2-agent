"""How a frame is grabbed, behind one interface for every platform.

Everything after capture (detection, the HUD reader, tracking) sees only a `Frame`.
Each platform supplies its own grabbers:

- Windows: `mss`, one region grab per call, exactly as before.
- macOS: a ScreenCaptureKit stream of the game window, or one Quartz window
  capture per call. Both live in `screen_macos`, which only macOS imports.

`AOE2_CAPTURE_METHOD` picks one: `auto` (the default: the stream on macOS, `mss`
elsewhere), `stream`, `window` or `mss`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import mss
import structlog
from PIL import Image

if TYPE_CHECKING:
    from .window import GameGeometry, Platform

log = structlog.stdlib.get_logger()

CaptureMethod = Literal["auto", "stream", "window", "mss"]
CAPTURE_METHODS: tuple[CaptureMethod, ...] = ("auto", "stream", "window", "mss")
_MAC_METHODS = ("stream", "window")


@dataclass(frozen=True, slots=True)
class Frame:
    """RGB pixels of the game area, and when the screen looked like this."""

    rgb: bytes
    width: int
    height: int
    captured_at: float  # time.monotonic()


class FrameGrabber(Protocol):
    def grab(
        self, geometry: GameGeometry | None, monitor: int, *, after: float | None = None
    ) -> Frame:
        """The game area, or the whole monitor when the window is unknown.

        With `after`, the frame shows the screen later than that `time.monotonic()`
        instant, so an input sent before it can be in the picture.
        """
        ...

    def close(self) -> None:
        """Release what the grabber holds. Never raises."""
        ...


class MssGrabber:
    """One `mss` grab per call: the Windows capture, unchanged."""

    def grab(
        self, geometry: GameGeometry | None, monitor: int, *, after: float | None = None
    ) -> Frame:
        del after  # Every grab is taken now.
        stamped = time.monotonic()
        with mss.MSS() as sct:
            if geometry is not None:
                left, top, width, height = geometry.rect
                shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
            else:
                shot = sct.grab(sct.monitors[monitor])
            image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return Frame(image.tobytes(), image.width, image.height, stamped)

    def close(self) -> None:
        return None


def resolve_method(method: str, platform: Platform) -> CaptureMethod:
    """The concrete method for this platform; raises on one the platform lacks."""
    if method == "auto":
        return "stream" if platform == "macos" else "mss"
    supported = _MAC_METHODS if platform == "macos" else ("mss",)
    if method not in supported:
        raise ValueError(
            f"capture method {method!r} doesn't work on {platform}; use one of "
            f"{', '.join(('auto', *supported))}"
        )
    return method  # pyright: ignore[reportReturnType]  # Narrowed by `supported`.


def make_grabber(method: str, platform: Platform) -> FrameGrabber:
    """Build the grabber. With `auto` on macOS, a stream that won't start falls back
    to window capture, which is slower but needs nothing ScreenCaptureKit needs."""
    resolved = resolve_method(method, platform)
    if resolved == "mss":
        return MssGrabber()
    from .screen_macos import StreamGrabber, WindowGrabber

    if resolved == "window":
        return WindowGrabber()
    if method == "auto":
        return FallbackGrabber(StreamGrabber(), WindowGrabber())
    return StreamGrabber()


class FallbackGrabber:
    """Use `primary` until it fails once, then `fallback` for the rest of the run."""

    def __init__(self, primary: FrameGrabber, fallback: FrameGrabber) -> None:
        self._primary: FrameGrabber | None = primary
        self._fallback = fallback

    def grab(
        self, geometry: GameGeometry | None, monitor: int, *, after: float | None = None
    ) -> Frame:
        primary = self._primary
        if primary is not None:
            try:
                return primary.grab(geometry, monitor, after=after)
            except Exception as error:
                log.warning(
                    "capture_fell_back",
                    primary=type(primary).__name__,
                    fallback=type(self._fallback).__name__,
                    error=f"{type(error).__name__}: {error}",
                )
                primary.close()
                self._primary = None
        return self._fallback.grab(geometry, monitor, after=after)

    def close(self) -> None:
        if self._primary is not None:
            self._primary.close()
        self._fallback.close()


__all__ = [
    "CAPTURE_METHODS",
    "CaptureMethod",
    "FallbackGrabber",
    "Frame",
    "FrameGrabber",
    "MssGrabber",
    "make_grabber",
    "resolve_method",
]
