"""Capture the native Mac game at full Retina resolution.

The capture names the game's window, so another app's window on top can't leak into
the frame. It is clipped to the game area, without the band under the menu bar.
"""

# pyobjc fills its framework modules at runtime, so a static checker can't see
# Quartz's functions and constants.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import Quartz

if TYPE_CHECKING:
    from .window import GameGeometry

# Best resolution gives pixels, not points; without it the frame comes back half size.
IMAGE_OPTIONS = (
    Quartz.kCGWindowImageBoundsIgnoreFraming
    | Quartz.kCGWindowImageShouldBeOpaque
    | Quartz.kCGWindowImageBestResolution
)


def bgra_to_rgb(raw: bytes, width: int, height: int, row_bytes: int) -> np.ndarray:
    """RGB pixels from Quartz's BGRA rows, which may be padded past the image width."""
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes // 4, 4)[:, :width]
    return np.ascontiguousarray(pixels[..., 2::-1])


def grab_game_area(geometry: GameGeometry) -> np.ndarray:
    """The game area as RGB pixels, exactly `geometry.width` by `geometry.height`."""
    if geometry.window_id is None:
        raise ValueError("a Mac capture needs the game's window number")
    scale = geometry.scale
    area = Quartz.CGRectMake(
        geometry.left / scale,
        geometry.top / scale,
        geometry.width / scale,
        geometry.height / scale,
    )
    image = Quartz.CGWindowListCreateImage(
        area, Quartz.kCGWindowListOptionIncludingWindow, geometry.window_id, IMAGE_OPTIONS
    )
    if image is None:
        raise RuntimeError(
            "the window capture returned nothing; check that this terminal has "
            "Screen Recording permission"
        )
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    if (width, height) != (geometry.width, geometry.height):
        raise RuntimeError(
            f"captured {width}x{height}, but the game area is {geometry.width}x{geometry.height}"
        )
    raw = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image)))
    return bgra_to_rgb(raw, width, height, Quartz.CGImageGetBytesPerRow(image))
