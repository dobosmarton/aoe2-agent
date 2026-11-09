"""The tracker's appearance signature for live frames, computed fast.

`temporal_entities.prepare_detections` takes a NumPy mean and standard deviation
of every box at full resolution. On a 3024x1898 Mac frame with 34 boxes that cost
390 ms, and with the game running it competed with the detector for the CPU.
OpenCV's `meanStdDev` gives the same numbers (largest difference 8e-14) in 9 ms.

`temporal_entities.py` is left as it is: the match 4 tracker test pins its exact
blob, and offline tools keep using it. A test holds the two equal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2

from .temporal_entities import PreparedDetection

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from .entity_snapshot import EntitySnapshot

_EMPTY = (0.0,) * 6


def appearance(pixels: np.ndarray, box: tuple[float, ...]) -> tuple[float, ...]:
    """Per-channel mean and standard deviation of the box, each over 255."""
    height, width = cast("tuple[int, int]", pixels.shape[:2])
    x0, y0, x1, y1 = (int(value) for value in box)
    crop = pixels[max(0, y0) : max(0, min(height, y1)), max(0, x0) : max(0, min(width, x1))]
    if crop.size == 0:
        return _EMPTY
    mean, std = cast("tuple[np.ndarray, np.ndarray]", cv2.meanStdDev(crop))
    return (
        *cast("list[float]", (mean.ravel() / 255.0).tolist()),
        *cast("list[float]", (std.ravel() / 255.0).tolist()),
    )


def prepare_detections(
    entities: Sequence[EntitySnapshot], pixels: np.ndarray
) -> tuple[PreparedDetection, ...]:
    """`temporal_entities.prepare_detections`, with the fast signature."""
    return tuple(
        PreparedDetection(item, appearance(pixels, item.bbox))
        for item in entities
        if item.bbox is not None
    )


__all__ = ["appearance", "prepare_detections"]
