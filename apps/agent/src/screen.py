"""Screen capture module for AoE2 LLM Agent."""

import atexit
import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .capture import FrameGrabber, make_grabber
from .config import config
from .window import (
    GameGeometry,
    game_platform,
    get_game_geometry,
    get_input_scale,
)

RESOURCE_BAR_HEIGHT = 60
RESOURCE_BAR_JPEG_QUALITY = 80


@dataclass(frozen=True, slots=True)
class NativeCapture:
    """One uncompressed window capture; encodings derive from these pixels."""

    rgb: bytes
    width: int
    height: int
    captured_at: float
    window_rect: tuple[int, int, int, int] | None
    cursor: tuple[int, int] | None = None  # Window-relative, sampled near capture.

    def image(self) -> Image.Image:
        return Image.frombytes("RGB", (self.width, self.height), self.rgb)

    def pixels(self) -> np.ndarray:
        """Read-only RGB view backed by this capture's immutable bytes."""
        return np.frombuffer(self.rgb, dtype=np.uint8).reshape((self.height, self.width, 3))

    def hud_pixels(self) -> np.ndarray:
        return self.pixels()[: min(self.height, 300)]

    def jpeg(self, quality: int | None = None) -> bytes:
        return _jpeg(self.image(), config.screenshot_quality if quality is None else quality)


_grabber: FrameGrabber | None = None
_grabber_lock = threading.Lock()


def frame_grabber() -> FrameGrabber:
    """The capture method for this run, built on first use (see `capture.py`)."""
    global _grabber
    with _grabber_lock:
        if _grabber is None:
            _grabber = make_grabber(config.capture_method, game_platform())
        return _grabber


def close_capture() -> None:
    """Stop the capture method, such as the Mac stream. The next grab starts it again."""
    global _grabber
    with _grabber_lock:
        grabber, _grabber = _grabber, None
    if grabber is not None:
        grabber.close()


atexit.register(close_capture)


@contextmanager
def use_grabber(grabber: FrameGrabber) -> Iterator[FrameGrabber]:
    """Capture through `grabber` inside the block; for tests and tools."""
    global _grabber
    with _grabber_lock:
        previous, _grabber = _grabber, grabber
    try:
        yield grabber
    finally:
        with _grabber_lock:
            _grabber = previous


def _game_geometry() -> GameGeometry | None:
    """The game's geometry. On macOS there is no capture without the window: the whole
    display would show this terminal, not the game."""
    geometry = get_game_geometry()
    if geometry is None and game_platform() == "macos":
        raise RuntimeError("the game window wasn't found, so nothing was captured")
    return geometry


def _capture_image(monitor: int) -> Image.Image:
    """Read native game-window pixels once, before either encoding path."""
    frame = frame_grabber().grab(_game_geometry(), monitor)
    return Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)


def capture_native(monitor: int = 1) -> NativeCapture:
    """Keep native pixels so detection, HUD and camera estimation share a view."""
    geometry = _game_geometry()
    frame = frame_grabber().grab(geometry, monitor)
    rect = geometry.rect if geometry is not None else None
    return NativeCapture(
        frame.rgb, frame.width, frame.height, frame.captured_at, rect, _cursor_in(rect)
    )


def _cursor_in(rect: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
    """The cursor in frame pixels, relative to the window. pyautogui reports input units."""
    if rect is None:
        return None
    try:
        import pyautogui

        position = pyautogui.position()
        scale = get_input_scale()
        return (round(position.x * scale) - rect[0], round(position.y * scale) - rect[1])
    except Exception:
        # Cursor masking is diagnostic enrichment, never a reason to lose
        # the native frame if a desktop automation backend is unavailable.
        return None


def _jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def capture_screenshot(monitor: int = 1, quality: int | None = None) -> tuple[bytes, int, int]:
    """
    Capture the game window and return as JPEG bytes with dimensions.

    Tries to capture only the game window region. Falls back to full monitor
    if window not found.

    Args:
        monitor: Monitor index (1 = primary monitor, used as fallback)
        quality: JPEG quality (1-100). Defaults to config.screenshot_quality.

    Returns:
        Tuple of (JPEG image bytes, width, height)
    """
    if quality is None:
        quality = config.screenshot_quality

    image = _capture_image(monitor)
    return _jpeg(image, quality), image.width, image.height


def save_screenshot(data: bytes, path: str) -> None:
    """Save screenshot bytes to a file."""
    with Path(path).open("wb") as f:
        f.write(data)


def crop_resource_bar(
    screenshot_bytes: bytes,
    bar_height: int = RESOURCE_BAR_HEIGHT,
) -> bytes:
    """Crop just the resource bar from the top of a screenshot.

    Args:
        screenshot_bytes: Full JPEG screenshot bytes
        bar_height: Height of the resource bar crop in pixels

    Returns:
        JPEG bytes of the cropped resource bar
    """
    img = Image.open(io.BytesIO(screenshot_bytes))
    bar = img.crop((0, 0, img.width, min(bar_height, img.height)))
    buffer = io.BytesIO()
    bar.save(buffer, format="JPEG", quality=RESOURCE_BAR_JPEG_QUALITY)
    return buffer.getvalue()


def capture_and_save(path: str, monitor: int = 1) -> tuple[bytes, int, int]:
    """Capture screenshot and save to file, returning (bytes, width, height)."""
    data, width, height = capture_screenshot(monitor)
    save_screenshot(data, path)
    return data, width, height
