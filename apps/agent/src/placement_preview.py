"""The game's own verdict on a building spot.

With a building on the cursor, the game draws it at the cursor in its normal
colours where it can go and tinted red where it can't. Measured on a Mac with a
house: the strong-red share around the spot stayed at 0.000 on open ground, and
rose to 0.053 over a tree and 0.177 over the Town Center. Before the hover check
the agent judged spots from its own detections, which miss bushes, snow-covered
trees and unexplored ground, and it learned of a refusal only 6 s after the click.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A preview pixel is strongly red: a high red channel, low green and blue.
_RED_MIN = 170
_OTHER_MAX = 90
# The red share must rise by this much over the frame without a preview.
_REFUSED_RISE = 0.01
_CHANGED_LEVEL = 40  # A pixel changed if any channel moved by more than this.


@dataclass(frozen=True, slots=True)
class PreviewRegion:
    """Where the preview of a building placed at one point is drawn, in frame pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def around(
        cls, point: tuple[int, int], half_width: float, half_height: float, size: tuple[int, int]
    ) -> PreviewRegion:
        """The footprint, and the sprite above it: a building stands as tall as it is wide."""
        width, height = size
        x, y = point
        return cls(
            x0=max(0, int(x - half_width)),
            y0=max(0, int(y - 2 * half_width)),
            x1=min(width, int(x + half_width)),
            y1=min(height, int(y + half_height)),
        )

    def crop(self, pixels: np.ndarray) -> np.ndarray:
        return pixels[self.y0 : self.y1, self.x0 : self.x1]


def red_share(pixels: np.ndarray, region: PreviewRegion) -> float:
    """The share of strongly red pixels in `region`."""
    crop = region.crop(pixels).astype(np.int16)
    if crop.size == 0:
        return 0.0
    red, green, blue = crop[..., 0], crop[..., 1], crop[..., 2]
    strong = (red >= _RED_MIN) & (green <= _OTHER_MAX) & (blue <= _OTHER_MAX)
    return float(strong.mean())


def changed_share(without: np.ndarray, with_preview: np.ndarray, region: PreviewRegion) -> float:
    """The share of `region` a preview changed, for the run log.

    Measured with a house: 0.22-0.44 with it on the cursor, 0.000-0.001 with
    nothing there. Not used to decide yet: farms and large buildings are unmeasured.
    """
    before = region.crop(without).astype(np.int16)
    after = region.crop(with_preview).astype(np.int16)
    if before.size == 0 or before.shape != after.shape:
        return 0.0
    return float((np.abs(after - before).max(axis=2) > _CHANGED_LEVEL).mean())


@dataclass(frozen=True, slots=True)
class PreviewReading:
    """What one hover showed at a candidate spot."""

    red_before: float
    red: float
    changed: float

    @property
    def refused(self) -> bool:
        """Whether the preview turned red.

        Comparing against the frame without a preview keeps red that was already
        there, such as a red player's units, from reading as a refusal.
        """
        return self.red - self.red_before >= _REFUSED_RISE


def read_preview(
    without: np.ndarray, with_preview: np.ndarray, region: PreviewRegion
) -> PreviewReading:
    """The preview at `region`, against a frame without one."""
    return PreviewReading(
        red_before=red_share(without, region),
        red=red_share(with_preview, region),
        changed=changed_share(without, with_preview, region),
    )


def preview_refused(without: np.ndarray, with_preview: np.ndarray, region: PreviewRegion) -> bool:
    """Whether the preview at `region` turned red, against a frame without a preview."""
    return read_preview(without, with_preview, region).refused


__all__ = [
    "PreviewReading",
    "PreviewRegion",
    "changed_share",
    "preview_refused",
    "read_preview",
    "red_share",
]
