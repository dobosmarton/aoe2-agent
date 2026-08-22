"""Frame differencing for skipping redundant rescans.

Compares consecutive screenshots at low resolution to answer two questions:
did the view change, and — when it did — how far did the camera pan. A pure
pan lets the caller translate cached static entities instead of re-detecting.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# One warning per process when opencv is absent, not one per rescan.
_cv2_warned = False

# Exclude top resource bar and bottom minimap area from comparison
_TOP_CROP_RATIO = 0.04  # ~4% of height (resource bar)
_BOTTOM_CROP_RATIO = 0.0  # minimap is in a corner, usually fine

# Downscale resolution for fast comparison
_COMPARE_WIDTH = 320
_COMPARE_HEIGHT = 180


@dataclass(frozen=True, slots=True)
class FrameChange:
    """How the view moved between two screenshots.

    `shift` is the camera pan in FULL-resolution pixels; `response` is phase
    correlation's confidence in it, 0 when there is none to give.
    """

    changed: bool
    shift: tuple[float, float] = (0.0, 0.0)
    response: float = 0.0


class FrameDiffer:
    """Detects visual changes between consecutive game screenshots.

    Usage:
        differ = FrameDiffer(threshold=0.03)

        # On each rescan:
        change = differ.compare(screenshot_bytes)
        if not change.changed:
            return  # Reuse previous entities
        if change.response >= 0.7:
            entities = translate(previous, change.shift)  # camera only panned
        else:
            entities = detector.detect_fast(screenshot_bytes)
    """

    def __init__(self, threshold: float = 0.03) -> None:
        """Initialize frame differ.

        Args:
            threshold: Mean Absolute Difference threshold (0-1 range).
                       0.03 = 3% average pixel change triggers re-detection.
                       Lower = more sensitive (fewer skips).
        """
        self.threshold = threshold
        self._prev_frame: np.ndarray | None = None
        # Downscale factors, so a shift measured on the compare array converts
        # back to screen pixels. Learned from the first screenshot seen.
        self._x_scale = 1.0
        self._y_scale = 1.0

    def _to_compare_array(self, screenshot: bytes) -> np.ndarray:
        """The downscaled grayscale play area, and the scale it was reduced by.

        Recording `_x_scale` / `_y_scale` here is what lets `_pan_between`
        convert a shift measured on this array back to screen pixels.
        """
        from PIL import Image

        img = Image.open(io.BytesIO(screenshot))
        w, h = img.size

        # Crop out resource bar at top
        top = int(h * _TOP_CROP_RATIO)
        bottom = int(h * (1 - _BOTTOM_CROP_RATIO))
        img = img.crop((0, top, w, bottom))

        self._x_scale = img.width / _COMPARE_WIDTH
        self._y_scale = img.height / _COMPARE_HEIGHT

        # Downscale and convert to grayscale
        img = img.resize((_COMPARE_WIDTH, _COMPARE_HEIGHT)).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def compare(self, screenshot: bytes) -> FrameChange:
        """Whether the view changed, and how far the camera panned if it did.

        Both answers in one call: reading the stored frame and replacing it are
        the same step, so two calls would race over it.
        """
        current = self._to_compare_array(screenshot)
        previous = self._prev_frame
        self._prev_frame = current
        if previous is None:
            return FrameChange(changed=True)
        if np.mean(np.abs(current - previous)) <= self.threshold:
            return FrameChange(changed=False)
        (dx, dy), response = self._pan_between(previous, current)
        return FrameChange(changed=True, shift=(dx, dy), response=response)

    def _pan_between(
        self, previous: np.ndarray, current: np.ndarray
    ) -> tuple[tuple[float, float], float]:
        """Camera pan in full-resolution pixels, with its confidence.

        Deferred import because this package does not declare opencv — the
        fallback is the contract, not defensiveness. It is announced rather than
        silent: losing the pan silently loses the whole rescan cache.
        """
        try:
            import cv2
        except ImportError:
            global _cv2_warned
            if not _cv2_warned:
                _cv2_warned = True
                logger.warning("opencv missing — rescans cannot reuse a panned map")
            return (0.0, 0.0), 0.0
        (dx, dy), response = cv2.phaseCorrelate(previous, current)
        return (dx * self._x_scale, dy * self._y_scale), float(response)

    def update(self, screenshot: bytes) -> None:
        """Force-update the stored frame (call after successful detection)."""
        self._prev_frame = self._to_compare_array(screenshot)

    def reset(self) -> None:
        """Clear stored frame (e.g., on camera movement)."""
        self._prev_frame = None
