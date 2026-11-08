"""Decode YOLO26 ONNX output tensors into typed detection rows.

A single source of truth shared by every inference path: the single-image path
(`detector._onnx_detect`), the batched-SAHI tile path (`sahi.parse_onnx_tile`)
and the detection server (`detection_server.app._parse_raw_output`), so they can
never disagree about how to read a tensor.

Two export layouts are accepted, one example (image or tile) at a time:

- End-to-end (``end2end=True``, the Ultralytics default): ``(num_boxes, 6)`` →
  ``[x1, y1, x2, y2, conf, class]``, already free of duplicates.
- One-to-many (``end2end=False``): ``(4 + num_classes, num_anchors)`` → rows of
  ``[xc, yc, w, h]`` and one score per class. Each anchor keeps its best class,
  and a class-wise NMS at `RAW_NMS_IOU` removes duplicates here, so callers get
  the same kind of rows from both layouts. On the independent test set this head
  scored F1 0.876 against 0.867 for the end-to-end head of the same weights.

Any other shape raises `UnknownOnnxLayoutError`, so an unexpected export fails
loudly instead of being silently mis-parsed. Coordinates are returned in
*model-input* pixels; each caller applies its own scale-back and tile offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

# Same IoU as the client-side `postprocess.nms`, so a second pass changes nothing.
RAW_NMS_IOU = 0.5


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
    """Raised when an ONNX output example matches neither YOLO26 layout."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__(
            f"Unrecognised ONNX output shape {shape}; expected (num_boxes, 6) end-to-end "
            "or (4 + num_classes, num_anchors) one-to-many."
        )


def decode_example(
    example: np.ndarray, min_confidence: float, num_classes: int | None = None
) -> list[DetectionRow]:
    """Decode one image's or tile's YOLO26 output into detection rows.

    `example` is the 2D output for one batch entry (the caller selects it from
    the batch dimension). The one-to-many layout is recognised only when the
    caller passes `num_classes`. Rows below `min_confidence` are dropped here;
    per-class thresholds remain the caller's responsibility.
    """
    if example.ndim == 2 and example.shape[1] == 6:
        rows = cast("list[list[float]]", example.tolist())
        return [
            DetectionRow(r[0], r[1], r[2], r[3], r[4], int(r[5]))
            for r in rows
            if r[4] >= min_confidence
        ]
    if num_classes is not None and example.ndim == 2 and example.shape[0] == 4 + num_classes:
        return _decode_one_to_many(example, min_confidence)
    raise UnknownOnnxLayoutError(example.shape)


def _decode_one_to_many(example: np.ndarray, min_confidence: float) -> list[DetectionRow]:
    """Best class per anchor, confidence filter, xywh → xyxy, class-wise NMS."""
    scores = example[4:]
    class_ids = scores.argmax(axis=0)
    confidences = scores.max(axis=0)
    keep = confidences >= min_confidence
    xc, yc, w, h = (example[i][keep] for i in range(4))
    boxes = np.stack((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), axis=1)
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    order = _class_wise_nms(boxes, confidences, class_ids, RAW_NMS_IOU)
    box_rows = cast("list[list[float]]", boxes[order].tolist())
    conf_list = cast("list[float]", confidences[order].tolist())
    cls_list = cast("list[int]", class_ids[order].tolist())
    return [
        DetectionRow(b[0], b[1], b[2], b[3], conf, cls)
        for b, conf, cls in zip(box_rows, conf_list, cls_list, strict=True)
    ]


def _class_wise_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Indices kept by greedy NMS, highest score first; classes never suppress each other.

    Matches `postprocess.nms`: a box is dropped when its IoU with a kept box of
    the same class is at least `iou_threshold`. Each class is moved to its own
    region of the plane so one pass over all boxes is class-wise.
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    offset = class_ids[:, None] * (float(np.abs(boxes).max()) + 1.0) * 2.0
    shifted = boxes + offset
    x1, y1, x2, y2 = shifted.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort(kind="stable")[::-1]
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        inter_w = np.clip(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0, None)
        inter_h = np.clip(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0, None)
        inter = inter_w * inter_h
        union = areas[best] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap < iou_threshold]
    return np.array(keep, dtype=np.int64)
