"""Unit tests for the YOLO26 NMS-free ONNX output decoder.

Pure-numpy tests — no model weights, no `ultralytics`. They pin the contract that
`detector._onnx_detect` and `sahi.parse_onnx_tile` both depend on: the YOLO26
``(num_boxes, 6)`` layout decodes correctly, and anything else raises.
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


class TestUnknownLayout:
    """Anything that is not the (num_boxes, 6) layout must raise, not be guessed at."""

    @pytest.mark.parametrize(
        "shape",
        [(3, 7), (5,), (1, 10, 6)],
        ids=["wrong-cols", "one-dim", "three-dim"],
    )
    def test_raises_on_unrecognised_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(UnknownOnnxLayoutError):
            decode_example(np.zeros(shape), min_confidence=0.25)
