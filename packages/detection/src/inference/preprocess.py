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

    chw: np.ndarray  # (3, target, target) float32 in [0, 1]
    scale: float  # uniform resize factor applied before padding
    pad_x: int  # left padding in model-space pixels
    pad_y: int  # top padding in model-space pixels


def letterbox(image: Image.Image, target_size: int) -> Letterboxed:
    """Aspect-preserving resize of ``image`` into a ``target_size`` square.

    Scales by ``min(target/w, target/h)`` and centres the result on a grey
    ``PAD_COLOR`` canvas — matching Ultralytics' training preprocessing. Returns the
    CHW float32 tensor and the scale/pad needed to map detections back (see
    ``Letterboxed``).
    """
    from PIL import Image as PILImage

    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2

    canvas = PILImage.new("RGB", (target_size, target_size), PAD_COLOR)
    canvas.paste(image.resize((new_w, new_h)), (pad_x, pad_y))
    chw = np.transpose(np.array(canvas, dtype=np.float32) / 255.0, (2, 0, 1))
    return Letterboxed(chw=chw, scale=scale, pad_x=pad_x, pad_y=pad_y)
