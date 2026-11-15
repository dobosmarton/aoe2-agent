"""The game's own verdict on a building spot.

With a building on the cursor, the game draws its preview at the cursor: red
where it can't go. Most buildings draw their whole sprite, tinted red when
refused. A farm draws only the outline of its footprint, red when refused. The
agent judged spots from its own detections before, which miss bushes,
snow-covered trees and unexplored ground, and it learned of a refusal only 6 s
after the click.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# A preview pixel is strongly red: a high red channel, low green and blue.
_RED_MIN = 170
_OTHER_MAX = 90
_CHANGED_LEVEL = 40  # A pixel changed if any channel moved by more than this.
# The footprint region reaches a little past the diamond, so its outline is inside.
_OUTLINE_MARGIN = 1.15


class PreviewKind(Enum):
    """How the game draws a building's preview, which decides where and how to read it."""

    SPRITE = "sprite"  # The whole building, tinted red when refused.
    OUTLINE = "outline"  # Only the footprint's outline, drawn red when refused.


@dataclass(frozen=True, slots=True)
class _Calibration:
    """Thresholds for one preview kind, measured in a live Mac game."""

    refused_rise: float  # The red share must rise by this much to refuse.
    drawn_share: float  # The share that must change to prove a preview is there.


_CALIBRATION: dict[PreviewKind, _Calibration] = {
    # A house: red 0.000 on open ground, 0.053-0.177 over a tree or the Town Center;
    # 0.22-0.44 of the region changed with it on the cursor, 0.000-0.001 without.
    # Markets with nothing on the cursor changed 0.006-0.034 and were clicked anyway.
    PreviewKind.SPRITE: _Calibration(refused_rise=0.01, drawn_share=0.08),
    # A farm, over its footprint: a red outline read 0.0023-0.0204 at six blocked
    # spots, and every farm placed in game 11 read 0.000. An outline changes too
    # few pixels to prove it is drawn, so that check is off.
    PreviewKind.OUTLINE: _Calibration(refused_rise=0.001, drawn_share=0.0),
}


@dataclass(frozen=True, slots=True)
class PreviewRegion:
    """Where a building's preview is read for one point, in frame pixels."""

    x0: int
    y0: int
    x1: int
    y1: int
    kind: PreviewKind = PreviewKind.SPRITE

    @classmethod
    def around(
        cls,
        point: tuple[int, int],
        half_width: float,
        half_height: float,
        size: tuple[int, int],
        kind: PreviewKind = PreviewKind.SPRITE,
    ) -> PreviewRegion:
        """A sprite's footprint and the building above it, or an outline's footprint alone.

        A building stands about as tall as it is wide. A farm's outline is a thin
        line; read over the sprite's height it was diluted below any threshold.
        """
        width, height = size
        x, y = point
        if kind is PreviewKind.OUTLINE:
            reach_x = half_width * _OUTLINE_MARGIN
            above = below = half_height * _OUTLINE_MARGIN
        else:
            reach_x, above, below = half_width, 2 * half_width, half_height
        return cls(
            x0=max(0, int(x - reach_x)),
            y0=max(0, int(y - above)),
            x1=min(width, int(x + reach_x)),
            y1=min(height, int(y + below)),
            kind=kind,
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
    """The share of `region` that changed against a frame without a preview."""
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
    kind: PreviewKind = PreviewKind.SPRITE

    @property
    def refused(self) -> bool:
        """Whether the preview turned red.

        Comparing against the frame without a preview keeps red that was already
        there, such as a red player's units, from reading as a refusal.
        """
        return self.red - self.red_before >= _CALIBRATION[self.kind].refused_rise

    @property
    def drawn(self) -> bool:
        """Whether a preview visibly appeared, so that "not red" means something."""
        return self.changed >= _CALIBRATION[self.kind].drawn_share

    @property
    def accepted(self) -> bool:
        return self.drawn and not self.refused


def read_preview(
    without: np.ndarray, with_preview: np.ndarray, region: PreviewRegion
) -> PreviewReading:
    """The preview at `region`, against a frame without one."""
    return PreviewReading(
        red_before=red_share(without, region),
        red=red_share(with_preview, region),
        changed=changed_share(without, with_preview, region),
        kind=region.kind,
    )


def preview_refused(without: np.ndarray, with_preview: np.ndarray, region: PreviewRegion) -> bool:
    """Whether the preview at `region` turned red, against a frame without a preview."""
    return read_preview(without, with_preview, region).refused


__all__ = [
    "PreviewKind",
    "PreviewReading",
    "PreviewRegion",
    "changed_share",
    "preview_refused",
    "read_preview",
    "red_share",
]
