"""Detection post-processing: non-maximum suppression and IoU.

Pure functions — no detector state. Hoisted out of `EntityDetector` because
they reference no instance attributes; the `self` argument was incidental.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detector import DetectedEntity


def iou(box1: tuple, box2: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms(entities: list[DetectedEntity], iou_threshold: float = 0.5) -> list[DetectedEntity]:
    """Non-maximum suppression: keep highest-confidence box per overlapping cluster.

    Two entities collide only if they share a class — different classes are
    allowed to overlap. Within a class, the higher-confidence box wins and
    everything overlapping it above `iou_threshold` is dropped.
    """
    if not entities:
        return []

    entities = sorted(entities, key=lambda e: -e.confidence)

    keep: list[DetectedEntity] = []
    while entities:
        best = entities.pop(0)
        keep.append(best)
        entities = [
            e
            for e in entities
            if e.class_name != best.class_name or iou(best.bbox, e.bbox) < iou_threshold
        ]

    return keep
