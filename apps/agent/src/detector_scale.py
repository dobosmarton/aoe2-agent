"""Keep objects at the size the detector learned, on any screen.

The detector was trained on VM frames, and loses accuracy when objects come out
larger (enlarging the VM test frames 1.7x cost v11 0.075 F1). Resolution, render
scaling and zoom all change object size, so the agent measures it from the game
itself: every whole Town Center it sees, against the VM's.

When objects are more than 15% too large, frames are shrunk onto a canvas of the
same size before detection, and the boxes are scaled back. The detector server
letterboxes by the long side, so shrinking inside the canvas is what changes the
size the model sees. Objects that are too small are only reported: enlarging
would crop the edges of the view, and zooming in is the fix.
"""

from __future__ import annotations

import io
import statistics
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import structlog
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from .entity_snapshot import EntitySnapshot

log = structlog.stdlib.get_logger()

# Median Town Center label width on the VM test set (39 labels in
# tmp/test_data_independent_20261030, 3024x1672 frames): 392.2 px, as a share of
# the frame's long side, which is what the detector's letterbox keeps.
REFERENCE_TC_WIDTH = 392.2 / 3024
# Plan rule R2: object size counts as matched at 0.9-1.15x the VM's.
MATCHED_RANGE = (0.9, 1.15)
# Height over width of whole Town Centers: 0.63-0.75 in the VM test labels, and
# 0.70-0.78 in v11's boxes on Mac frames. Boxes outside this were cut by the
# screen edge or the HUD, so their width misleads.
TC_ASPECT_RANGE = (0.58, 0.85)
TC_CLASS = "town_center"
MIN_CONFIDENCE = 0.5
EDGE_MARGIN_PX = 4
WINDOW = 9  # Recent Town Center widths kept; zoom changes age out of it.
MIN_SAMPLES = 3
SCALE_STEP = 0.05  # Scales are rounded to this, so noise doesn't re-scale every frame.
PAD_COLOUR = (114, 114, 114)  # The letterbox grey the detector already sees.


def town_center_widths(
    entities: Iterable[EntitySnapshot], frame_size: tuple[int, int]
) -> list[float]:
    """Widths of whole, confident Town Centers, as shares of the frame's long side."""
    width, height = frame_size
    long_side = max(width, height)
    widths: list[float] = []
    for entity in entities:
        if entity.class_name != TC_CLASS or entity.bbox is None:
            continue
        if entity.confidence < MIN_CONFIDENCE:
            continue
        x0, y0, x1, y1 = entity.bbox
        box_width, box_height = x1 - x0, y1 - y0
        if box_width <= 0:
            continue
        if not TC_ASPECT_RANGE[0] <= box_height / box_width <= TC_ASPECT_RANGE[1]:
            continue
        if min(x0, y0) < EDGE_MARGIN_PX:
            continue
        if x1 > width - EDGE_MARGIN_PX or y1 > height - EDGE_MARGIN_PX:
            continue
        widths.append(box_width / long_side)
    return widths


def scale_for(ratio: float) -> float:
    """The shrink that brings objects back to the VM's size; 1.0 when none is needed."""
    if ratio <= MATCHED_RANGE[1]:
        return 1.0
    return round(round((1 / ratio) / SCALE_STEP) * SCALE_STEP, 2)


class DetectorScale:
    """A running measure of object size, and the shrink it calls for."""

    def __init__(self) -> None:
        self._widths: deque[float] = deque(maxlen=WINDOW)
        self.ratio: float | None = None
        self.scale = 1.0

    def observe(self, entities: Iterable[EntitySnapshot], frame_size: tuple[int, int]) -> None:
        """Add this frame's Town Centers. Boxes must be in frame pixels, not shrunk."""
        widths = town_center_widths(entities, frame_size)
        if not widths:
            return
        self._widths.extend(widths)
        if len(self._widths) < MIN_SAMPLES:
            return
        ratio = statistics.median(self._widths) / REFERENCE_TC_WIDTH
        scale = scale_for(ratio)
        changed = scale != self.scale
        was_measured = self.ratio is not None
        self.ratio, self.scale = ratio, scale
        if changed or not was_measured:
            self._report(ratio, scale)

    @staticmethod
    def _report(ratio: float, scale: float) -> None:
        if ratio < MATCHED_RANGE[0]:
            log.warning(
                "objects_smaller_than_detector_training",
                size_ratio=round(ratio, 2),
                advice="zoom in: frames are not enlarged",
            )
            return
        log.info("detector_scale_set", size_ratio=round(ratio, 2), frame_scale=scale)


def shrunk_jpeg(
    rgb: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]], scale: float, quality: int
) -> bytes:
    """The frame shrunk by `scale` into the top-left of a same-size grey canvas."""
    height, width, _ = rgb.shape
    image = Image.fromarray(rgb).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (width, height), PAD_COLOUR)
    canvas.paste(image, (0, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def restore(entities: Iterable[EntitySnapshot], scale: float) -> tuple[EntitySnapshot, ...]:
    """Boxes found on a shrunk frame, back in the frame's own pixels."""
    restored: list[EntitySnapshot] = []
    for entity in entities:
        bbox = entity.bbox
        if bbox is not None:
            bbox = (bbox[0] / scale, bbox[1] / scale, bbox[2] / scale, bbox[3] / scale)
        center = (entity.center[0] / scale, entity.center[1] / scale)
        restored.append(replace(entity, bbox=bbox, center=center))
    return tuple(restored)


__all__ = [
    "MATCHED_RANGE",
    "REFERENCE_TC_WIDTH",
    "DetectorScale",
    "restore",
    "scale_for",
    "shrunk_jpeg",
    "town_center_widths",
]
