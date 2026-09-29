"""Screen capture module for AoE2 LLM Agent."""

import io
from pathlib import Path

import mss
from PIL import Image

from .config import config
from .window import get_game_window_rect

RESOURCE_BAR_HEIGHT = 60
RESOURCE_BAR_JPEG_QUALITY = 80


def _capture_image(monitor: int) -> Image.Image:
    """Read native game-window pixels once, before either encoding path."""
    with mss.mss() as sct:
        rect = get_game_window_rect()
        if rect:
            left, top, width, height = rect
            region = {"left": left, "top": top, "width": width, "height": height}
            screenshot = sct.grab(region)
        else:
            screenshot = sct.grab(sct.monitors[monitor])
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


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


def capture_screenshot_with_native_hud(
    monitor: int = 1, quality: int | None = None
) -> tuple[bytes, bytes, int, int]:
    """Return full JPEG plus lossless top-HUD pixels from the same capture."""
    image = _capture_image(monitor)
    hud = image.crop((0, 0, image.width, min(image.height, 300)))
    hud_buffer = io.BytesIO()
    hud.save(hud_buffer, format="PNG")
    return (
        _jpeg(image, config.screenshot_quality if quality is None else quality),
        hud_buffer.getvalue(),
        image.width,
        image.height,
    )


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
