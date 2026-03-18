"""Frame differencing for skipping redundant rescans.

Compares consecutive screenshots at low resolution to detect whether
the game state has visually changed. If the change is below a threshold,
the rescan can be skipped and previous entity positions reused.
"""

from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Exclude top resource bar and bottom minimap area from comparison
_TOP_CROP_RATIO = 0.04    # ~4% of height (resource bar)
_BOTTOM_CROP_RATIO = 0.0  # minimap is in a corner, usually fine

# Downscale resolution for fast comparison
_COMPARE_WIDTH = 320
_COMPARE_HEIGHT = 180


class FrameDiffer:
    """Detects visual changes between consecutive game screenshots.

    Usage:
        differ = FrameDiffer(threshold=0.03)

        # On each rescan:
        if not differ.has_changed(screenshot_bytes):
            return  # Skip detection, reuse previous entities
        entities = detector.detect_fast(screenshot_bytes)
        differ.update(screenshot_bytes)
    """

    def __init__(self, threshold: float = 0.03):
        """Initialize frame differ.

        Args:
            threshold: Mean Absolute Difference threshold (0-1 range).
                       0.03 = 3% average pixel change triggers re-detection.
                       Lower = more sensitive (fewer skips).
        """
        self.threshold = threshold
        self._prev_frame: np.ndarray | None = None

    def _to_grayscale_array(self, screenshot: bytes) -> np.ndarray:
        """Convert screenshot bytes to a small grayscale numpy array."""
        from PIL import Image

        img = Image.open(io.BytesIO(screenshot))
        w, h = img.size

        # Crop out resource bar at top
        top = int(h * _TOP_CROP_RATIO)
        bottom = int(h * (1 - _BOTTOM_CROP_RATIO))
        img = img.crop((0, top, w, bottom))

        # Downscale and convert to grayscale
        img = img.resize((_COMPARE_WIDTH, _COMPARE_HEIGHT)).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def has_changed(self, screenshot: bytes) -> bool:
        """Check if the screenshot has changed significantly from the previous one.

        Returns True if:
        - No previous frame stored (first call)
        - Mean Absolute Difference exceeds threshold
        """
        current = self._to_grayscale_array(screenshot)

        if self._prev_frame is None:
            self._prev_frame = current
            return True  # First frame — always detect

        mad = np.mean(np.abs(current - self._prev_frame))
        changed = mad > self.threshold

        if changed:
            logger.debug("frame_diff changed=True mad=%.4f threshold=%.4f", mad, self.threshold)
            self._prev_frame = current
        else:
            logger.debug("frame_diff changed=False mad=%.4f threshold=%.4f", mad, self.threshold)

        return changed

    def update(self, screenshot: bytes) -> None:
        """Force-update the stored frame (call after successful detection)."""
        self._prev_frame = self._to_grayscale_array(screenshot)

    def reset(self) -> None:
        """Clear stored frame (e.g., on camera movement)."""
        self._prev_frame = None
