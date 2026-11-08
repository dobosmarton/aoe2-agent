"""Unit tests for the YOLO26 ONNX output decoder.

Pure-numpy tests — no model weights, no `ultralytics`. They pin the contract that
`detector._onnx_detect`, `sahi.parse_onnx_tile` and the detection server depend
on: the end-to-end ``(num_boxes, 6)`` layout and the one-to-many
``(4 + num_classes, num_anchors)`` layout decode correctly, and anything else
raises.
"""

from __future__ import annotations

import numpy as np
import pytest
from detection.inference.onnx_layout import (
    DetectionRow,
    UnknownOnnxLayoutError,
    decode_example,
)


class TestEndToEndLayout:
    """`(num_boxes, 6)` rows of [x1, y1, x2, y2, conf, class] (YOLO26 / NMS-free)."""

    def test_keeps_confident_rows_verbatim(self) -> None:
        example = np.array(
            [
                [10.0, 20.0, 30.0, 40.0, 0.90, 8.0],  # sheep, kept
                [5.0, 5.0, 15.0, 15.0, 0.40, 3.0],  # berry_bush, kept
            ]
        )

        rows = decode_example(example, min_confidence=0.25)

        assert rows == [
            DetectionRow(10.0, 20.0, 30.0, 40.0, 0.90, 8),
            DetectionRow(5.0, 5.0, 15.0, 15.0, 0.40, 3),
        ]

    def test_drops_rows_below_min_confidence(self) -> None:
        example = np.array(
            [
                [10.0, 20.0, 30.0, 40.0, 0.90, 8.0],  # kept
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zero-padded slot, dropped
                [1.0, 1.0, 2.0, 2.0, 0.10, 3.0],  # below threshold, dropped
            ]
        )

        rows = decode_example(example, min_confidence=0.25)

        assert len(rows) == 1
        assert rows[0].class_id == 8


def _one_to_many(anchors: list[tuple[float, float, float, float, list[float]]]) -> np.ndarray:
    """Build a (4 + nc, num_anchors) output from (xc, yc, w, h, class scores) anchors."""
    return np.array([[*box, *scores] for *box, scores in anchors], dtype=np.float32).T


class TestOneToManyLayout:
    """`(4 + nc, anchors)`: xywh plus one score per class, deduplicated by class-wise NMS."""

    def test_converts_xywh_and_keeps_the_best_class(self) -> None:
        example = _one_to_many([(50.0, 40.0, 20.0, 10.0, [0.1, 0.9, 0.2])])

        rows = decode_example(example, min_confidence=0.25, num_classes=3)

        assert len(rows) == 1
        row = rows[0]
        assert (row.x1, row.y1, row.x2, row.y2, row.class_id) == (40.0, 35.0, 60.0, 45.0, 1)
        assert row.confidence == pytest.approx(0.9)

    def test_nms_drops_same_class_duplicates_but_keeps_other_classes(self) -> None:
        example = _one_to_many(
            [
                (50.0, 50.0, 20.0, 20.0, [0.9, 0.0]),  # kept: best tree
                (51.0, 50.0, 20.0, 20.0, [0.8, 0.0]),  # dropped: IoU 0.9 with the first tree
                (50.0, 50.0, 20.0, 20.0, [0.0, 0.7]),  # kept: same box, other class
                (150.0, 50.0, 20.0, 20.0, [0.6, 0.0]),  # kept: a tree elsewhere
                (300.0, 50.0, 20.0, 20.0, [0.1, 0.1]),  # dropped: below min_confidence
            ]
        )

        rows = decode_example(example, min_confidence=0.25, num_classes=2)

        assert [(row.class_id, round(row.confidence, 2)) for row in rows] == [
            (0, 0.9),
            (1, 0.7),
            (0, 0.6),
        ]

    def test_no_anchor_above_the_threshold_gives_no_rows(self) -> None:
        example = _one_to_many([(50.0, 50.0, 20.0, 20.0, [0.1, 0.2])])

        assert decode_example(example, min_confidence=0.25, num_classes=2) == []

    def test_needs_the_class_count(self) -> None:
        example = _one_to_many([(50.0, 50.0, 20.0, 20.0, [0.9, 0.2, 0.1])])

        with pytest.raises(UnknownOnnxLayoutError):
            decode_example(example, min_confidence=0.25)


class TestUnknownLayout:
    """Anything that matches neither layout must raise, not be guessed at."""

    @pytest.mark.parametrize(
        "shape",
        [(3, 7), (5,), (1, 10, 6)],
        ids=["wrong-cols", "one-dim", "three-dim"],
    )
    def test_raises_on_unrecognised_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(UnknownOnnxLayoutError):
            decode_example(np.zeros(shape), min_confidence=0.25)
