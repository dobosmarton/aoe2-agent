"""Screen capture module for AoE2 LLM Agent."""

import io
import time
from dataclasses import dataclass
from pathlib import Path

import mss
import numpy as np
from PIL import Image

from .config import config
from .window import get_game_window_rect

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


def _capture_image(monitor: int, rect: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Read native game-window pixels once, before either encoding path."""
    with mss.mss() as sct:
        if rect is None:
            rect = get_game_window_rect()
        if rect:
            left, top, width, height = rect
            region = {"left": left, "top": top, "width": width, "height": height}
            screenshot = sct.grab(region)
        else:
            screenshot = sct.grab(sct.monitors[monitor])
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def capture_native(monitor: int = 1) -> NativeCapture:
    """Keep native pixels so detection, HUD and camera estimation share a view."""
    stamped = time.monotonic()
    rect = get_game_window_rect()
    image = _capture_image(monitor, rect)
    cursor: tuple[int, int] | None = None
    if rect is not None:
        try:
            import pyautogui

            position = pyautogui.position()
            cursor = (int(position.x) - rect[0], int(position.y) - rect[1])
        except Exception:
            # Cursor masking is diagnostic enrichment, never a reason to lose
            # the native frame if a desktop automation backend is unavailable.
            pass
    return NativeCapture(image.tobytes(), image.width, image.height, stamped, rect, cursor)


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
