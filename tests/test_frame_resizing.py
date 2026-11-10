"""Shrinking captured frames for the detector and for camera motion."""

from __future__ import annotations

import numpy as np
from detection.inference.preprocess import letterbox
from gameplay_agent.camera_motion import MotionLimits, analysis_pixels
from PIL import Image

MAC_W, MAC_H = 3024, 1898
MODEL_HW = (736, 1280)
# A resize that skips pixels turns black and white stripes into pure black or
# white. Averaging keeps every value well inside this band.
AVERAGED = (0.3, 0.7)


def test_an_array_and_an_image_letterbox_alike() -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (MAC_H, MAC_W, 3), np.uint8)
    from_array = letterbox(pixels, MODEL_HW)
    from_image = letterbox(Image.fromarray(pixels), MODEL_HW)
    assert np.array_equal(from_array.chw, from_image.chw)


def _stripes(width: int, height: int) -> np.ndarray:
    """Alternate black and white columns, which any pixel-skipping resize aliases."""
    stripes = np.zeros((height, width, 3), np.uint8)
    stripes[:, ::2] = 255
    return stripes


def test_shrinking_for_the_detector_averages_pixels() -> None:
    box = letterbox(_stripes(MAC_W, MAC_H), MODEL_HW)
    height, width = MODEL_HW
    image = box.chw[:, box.pad_y : height - box.pad_y, box.pad_x : width - box.pad_x]
    assert AVERAGED[0] < float(image.min()) and float(image.max()) < AVERAGED[1]


def _motion_view() -> np.ndarray:
    return analysis_pixels(_stripes(MAC_W, MAC_H).tobytes(), MAC_W, MAC_H, MotionLimits())


def test_camera_motion_gets_its_analysis_size() -> None:
    limits = MotionLimits()
    assert _motion_view().shape == (limits.analysis_height, limits.analysis_width)


def test_shrinking_for_camera_motion_averages_pixels() -> None:
    small = _motion_view()
    assert AVERAGED[0] < float(small.min()) and float(small.max()) < AVERAGED[1]
