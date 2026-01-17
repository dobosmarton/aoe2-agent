"""Screen capture module for AoE2 LLM Agent."""

import io

import mss
from PIL import Image

from .config import config
from .window import get_game_window_rect


def capture_screenshot(monitor: int = 1) -> tuple[bytes, int, int]:
    """
    Capture the game window and return as JPEG bytes with dimensions.

    Tries to capture only the game window region. Falls back to full monitor
    if window not found.

    Args:
        monitor: Monitor index (1 = primary monitor, used as fallback)

    Returns:
        Tuple of (JPEG image bytes, width, height)
    """
    with mss.mss() as sct:
        # Try to capture just the game window
        rect = get_game_window_rect()
        if rect:
            left, top, width, height = rect
            region = {"left": left, "top": top, "width": width, "height": height}
            screenshot = sct.grab(region)
        else:
            # Fall back to full monitor
            screenshot = sct.grab(sct.monitors[monitor])

        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=config.screenshot_quality)
        return buffer.getvalue(), img.width, img.height


def save_screenshot(data: bytes, path: str) -> None:
    """Save screenshot bytes to a file."""
    with open(path, "wb") as f:
        f.write(data)


def capture_and_save(path: str, monitor: int = 1) -> tuple[bytes, int, int]:
    """Capture screenshot and save to file, returning (bytes, width, height)."""
    data, width, height = capture_screenshot(monitor)
    save_screenshot(data, path)
    return data, width, height
