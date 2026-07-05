"""Unit tests for the shared YOLO letterbox preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from detection.inference.preprocess import PAD_COLOR, letterbox
from PIL import Image

# The agent's real capture resolution — the aspect ratio that exposed the squish bug.
SCREENSHOT_W, SCREENSHOT_H = 3024, 1672
TARGET = 1280


def _solid(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (10, 20, 30))


def test_letterbox_produces_normalized_square_chw() -> None:
    box = letterbox(_solid(SCREENSHOT_W, SCREENSHOT_H), TARGET)
    assert box.chw.shape == (3, TARGET, TARGET)
    assert box.chw.dtype == np.float32
    assert float(box.chw.min()) >= 0.0
    assert float(box.chw.max()) <= 1.0


def test_letterbox_preserves_aspect_with_symmetric_padding() -> None:
    box = letterbox(_solid(SCREENSHOT_W, SCREENSHOT_H), TARGET)
    # Width is the limiting axis, so it fills the square and only the height is padded.
    assert box.scale == pytest.approx(TARGET / SCREENSHOT_W)
    assert box.pad_x == 0
    assert box.pad_y == (TARGET - round(SCREENSHOT_H * box.scale)) // 2
    assert box.pad_y > 0


def test_letterbox_fills_padding_with_pad_color() -> None:
    box = letterbox(_solid(SCREENSHOT_W, SCREENSHOT_H), TARGET)
    top_strip = box.chw[:, 0, :]  # row 0 is padding (pad_y > 0)
    expected = np.array(PAD_COLOR, dtype=np.float32)[:, None] / 255.0
    assert np.allclose(top_strip, expected)


@pytest.mark.parametrize(
    ("orig_w", "orig_h"),
    [(3024, 1672), (1280, 720), (640, 640), (1000, 1600)],
    ids=["wide-native", "wide-16x9", "square", "tall"],
)
def test_image_center_maps_to_square_center(orig_w: int, orig_h: int) -> None:
    """Scale and pad are jointly correct: the image centre lands at the square centre,
    which is the invariant the inverse mapping in both detectors relies on."""
    box = letterbox(_solid(orig_w, orig_h), TARGET)
    center_x = (orig_w / 2) * box.scale + box.pad_x
    center_y = (orig_h / 2) * box.scale + box.pad_y
    assert center_x == pytest.approx(TARGET / 2, abs=1.0)
    assert center_y == pytest.approx(TARGET / 2, abs=1.0)
