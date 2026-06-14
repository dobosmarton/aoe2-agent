"""Decode YOLO26 NMS-free ONNX output tensors into typed detection rows.

A single source of truth shared by the two inference paths — the single-image
path (`detector._onnx_detect`) and the batched-SAHI tile path
(`sahi.parse_onnx_tile`) — so they can never disagree about how to read a tensor.

YOLO26 is end-to-end / NMS-free: each example is ``(num_boxes, 6)`` →
``[x1, y1, x2, y2, conf, class]`` in model-input pixels (xyxy). `decode_example`
parses exactly that and raises `UnknownOnnxLayoutError` on any other shape, so an
unexpected export fails loudly instead of being silently mis-parsed.

Coordinates are returned in *model-input* pixels; each caller applies its own
scale-back and tile offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """One decoded detection in model-input pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class UnknownOnnxLayoutError(ValueError):
    """Raised when an ONNX output example is not the YOLO26 (num_boxes, 6) layout."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__(
            f"Unrecognised ONNX output shape {shape}; expected (num_boxes, 6) end-to-end."
        )


def decode_example(example: np.ndarray, min_confidence: float) -> list[DetectionRow]:
    """Decode a single example's YOLO26 NMS-free output into detection rows.

    `example` is one image/tile's 2D ``(num_boxes, 6)`` output (the caller selects
    it from the batch dimension). Rows below `min_confidence` are dropped here;
    per-class thresholds remain the caller's responsibility.
    """
    if example.ndim != 2 or example.shape[1] != 6:
        raise UnknownOnnxLayoutError(example.shape)

    rows = cast("list[list[float]]", example.tolist())
    return [
        DetectionRow(r[0], r[1], r[2], r[3], r[4], int(r[5]))
        for r in rows
        if r[4] >= min_confidence
    ]
