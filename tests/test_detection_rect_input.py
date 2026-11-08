"""A static rectangular one-to-many export serves through the shared decoder.

The served v12 export is 736x1280 with the one-to-many head: a 16:9 frame
letterboxes at the same scale as into the 1280 square, and the server runs a
class-wise NMS on the raw anchors.
"""

from __future__ import annotations

import numpy as np
import pytest
from detection.inference.preprocess import letterbox
from detection_server import app as server
from PIL import Image


def test_letterbox_fits_a_wide_frame_into_a_rectangle_at_the_square_scale() -> None:
    frame = Image.new("RGB", (3024, 1672))

    square = letterbox(frame, 1280)
    rect = letterbox(frame, (736, 1280))

    assert rect.chw.shape == (3, 736, 1280)
    assert rect.scale == square.scale
    assert (rect.pad_x, rect.pad_y) == (0, 14)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ([1, 3, 736, 1280], (736, 1280)),
        (["batch", 3, "height", "width"], None),
        ([1, 3, 1280], None),
    ],
    ids=["static", "dynamic", "not-nchw"],
)
def test_static_input_shape(shape: list[object], expected: tuple[int, int] | None) -> None:
    assert server._static_input_hw(shape) == expected  # pyright: ignore[reportPrivateUsage]


class _RawSession:
    """Fake ONNX session: one-to-many output for 2 classes and 5 anchors."""

    def __init__(self) -> None:
        self.batch_shape: tuple[int, ...] = ()

    def run(self, _outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.batch_shape = feed["images"].shape
        anchors = [
            (640.0, 368.0, 100.0, 50.0, 0.9, 0.0),  # tree, kept
            (642.0, 368.0, 100.0, 50.0, 0.8, 0.0),  # duplicate tree, removed by NMS
            (200.0, 200.0, 40.0, 40.0, 0.0, 0.9),  # villager, kept
            (900.0, 300.0, 40.0, 40.0, 0.01, 0.02),  # below every threshold
            (100.0, 100.0, 10.0, 10.0, 0.0, 0.0),  # empty anchor
        ]
        return [np.array(anchors, dtype=np.float32).T[None]]


def test_detect_single_uses_the_static_shape_and_removes_duplicates() -> None:
    session = _RawSession()
    state = server.ModelState(
        backend="onnx_cpu",
        model=session,
        class_names=("tree", "villager"),
        model_path="model.onnx",
        input_name="images",
        input_hw=(736, 1280),
    )

    # 2560x1440 -> scale 0.5 to 1280x720, padded 8 px top and bottom.
    detections = server._detect_single(state, Image.new("RGB", (2560, 1440)), 1280)  # pyright: ignore[reportPrivateUsage]

    assert session.batch_shape == (1, 3, 736, 1280)
    by_class = {d.class_name: d for d in detections}
    assert len(detections) == 2
    assert set(by_class) == {"tree", "villager"}
    assert by_class["tree"].bbox == pytest.approx((1180.0, 670.0, 1380.0, 770.0))
