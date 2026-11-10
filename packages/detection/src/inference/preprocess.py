"""Letterbox preprocessing for YOLO inference.

Ultralytics trains and validates with *letterboxing* — an aspect-preserving resize
into a square with grey padding. Feeding the model a naively squished
``image.resize((n, n))`` distorts the 3024x1672 screenshot by ~1.81x and pushes it
out of the training distribution: small sprites (sheep) vanish and stretched
silhouettes get misclassified (a villager reads as ``battle_elephant``).

This module is the single home of that geometry, shared by the detection server and
the local ONNX detector so both preprocess identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# YOLO's standard letterbox fill. Also used to pad partial SAHI tiles so edge tiles
# match the grey-padded backgrounds the model saw during training.
PAD_COLOR: tuple[int, int, int] = (114, 114, 114)


class Letterboxed(NamedTuple):
    """A letterboxed image plus the transform needed to undo it.

    Recover an original coordinate from a model-space one, per axis, with
    ``orig = (coord - pad) / scale``.
    """

    chw: np.ndarray  # (3, target_h, target_w) float32 in [0, 1]
    scale: float  # uniform resize factor applied before padding
    pad_x: int  # left padding in model-space pixels
    pad_y: int  # top padding in model-space pixels


def letterbox(image: Image.Image | np.ndarray, target_size: int | tuple[int, int]) -> Letterboxed:
    """Aspect-preserving resize of ``image`` into the model input.

    ``image`` is a PIL image or an ``(H, W, 3)`` RGB array. ``target_size`` is a
    square side or a ``(height, width)`` pair: a static rectangular export such as
    736x1280 fits a 16:9 frame at the same scale as the 1280 square, with less
    padding to compute. Scales by ``min(target_w/w, target_h/h)`` and centres the
    result on a grey ``PAD_COLOR`` canvas, matching Ultralytics' training
    preprocessing. Returns the CHW float32 tensor and the scale/pad needed to map
    detections back (see ``Letterboxed``).

    The resize averages the source pixels each output pixel covers (OpenCV's area
    filter), as PIL's default resize did. It is about twice as fast on a 3024-wide
    Mac frame, and kept 99% of PIL's detections on 58 Mac frames, with the same
    validation F1. A bilinear resize was faster still, but skips pixels at that
    ratio and changed a tenth of the detections.
    """
    pixels = np.asarray(image)
    target_h, target_w = (target_size, target_size) if isinstance(target_size, int) else target_size
    orig_h, orig_w = pixels.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2

    canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
        pixels, (new_w, new_h), interpolation=cv2.INTER_AREA
    )
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return Letterboxed(chw=chw, scale=scale, pad_x=pad_x, pad_y=pad_y)
