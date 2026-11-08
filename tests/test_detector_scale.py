"""Object size is measured from Town Centers and corrected before detection."""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from core import DetectedEntity
from gameplay_agent.detector_scale import (
    MIN_SAMPLES,
    REFERENCE_TC_WIDTH,
    WINDOW,
    DetectorScale,
    restore,
    scale_for,
    shrunk_jpeg,
    town_center_widths,
)
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.loops.source import GameSource
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

_FRAME = (3024, 1898)


def _tc(
    width: float,
    *,
    x0: float = 1000,
    y0: float = 600,
    label: str = "town_center",
    confidence: float = 0.9,
    aspect: float = 0.67,
) -> EntitySnapshot:
    box = (x0, y0, x0 + width, y0 + width * aspect)
    return EntitySnapshot(
        "raw", label, box, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), confidence
    )


def _sized(ratio: float) -> EntitySnapshot:
    """A Town Center `ratio` times the VM's size, in a 3024-wide frame."""
    return _tc(REFERENCE_TC_WIDTH * 3024 * ratio)


def test_widths_are_shares_of_the_long_side() -> None:
    assert town_center_widths([_tc(392.2)], _FRAME) == [pytest.approx(REFERENCE_TC_WIDTH)]


@pytest.mark.parametrize(
    "entity",
    [
        _tc(392, label="house"),
        _tc(392, confidence=0.3),
        _tc(392, x0=2),  # Cut by the left edge.
        _tc(392, x0=2800),  # Runs off the right edge.
        _tc(392, aspect=1.0),  # Cut at the side: too tall for its width.
        _tc(392, aspect=0.4),  # Cut by the HUD: too flat.
    ],
)
def test_partial_or_doubtful_town_centers_are_not_measured(entity: EntitySnapshot) -> None:
    assert town_center_widths([entity], _FRAME) == []


@pytest.mark.parametrize(
    ("ratio", "scale"),
    [(0.77, 1.0), (0.96, 1.0), (1.10, 1.0), (1.15, 1.0), (1.23, 0.8), (1.8, 0.55)],
)
def test_only_objects_too_large_are_shrunk(ratio: float, scale: float) -> None:
    assert scale_for(ratio) == scale


def test_scale_waits_for_enough_town_centers() -> None:
    scale = DetectorScale()
    for _ in range(MIN_SAMPLES - 1):
        scale.observe([_sized(1.5)], _FRAME)
    assert (scale.ratio, scale.scale) == (None, 1.0)

    scale.observe([_sized(1.5)], _FRAME)
    assert scale.ratio == pytest.approx(1.5)
    assert scale.scale == 0.65


def test_a_zoom_change_ages_out_of_the_measure() -> None:
    scale = DetectorScale()
    for _ in range(WINDOW):
        scale.observe([_sized(1.5)], _FRAME)
    for _ in range(WINDOW // 2 + 1):
        scale.observe([_sized(1.0)], _FRAME)
    assert scale.scale == 1.0


def _open(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("L")


def _box_where(image: Image.Image, keep: Callable[[int], bool]) -> tuple[int, int, int, int]:
    """The bounding box of the pixels `keep` accepts."""
    box = image.point([255 if keep(value) else 0 for value in range(256)]).getbbox()
    assert box is not None
    return box


def test_shrunk_frame_keeps_its_size_and_restored_boxes_land_back() -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[100:200, 300:500] = 255
    shrunk = _open(shrunk_jpeg(frame, 0.5, quality=95))

    assert shrunk.size == (600, 400)
    assert _box_where(shrunk, lambda v: abs(v - 114) <= 3)[2:] == (600, 400)  # Canvas.
    found = EntitySnapshot("raw", "house", _box_where(shrunk, lambda v: v > 200), (0, 0), 0.9)
    (back,) = restore([found], 0.5)
    assert back.bbox == pytest.approx((300, 100, 500, 200), abs=3)


class _Detector:
    """Answers with one Town Center the given share of the image it is sent."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self.sizes: list[float] = []

    def detect_fast(self, image: bytes) -> list[DetectedEntity]:
        frame = _open(image)
        content = _box_where(frame, lambda v: abs(v - 114) > 20)[2]
        self.sizes.append(content / frame.width)
        width = REFERENCE_TC_WIDTH * frame.width * self.ratio * self.sizes[-1]
        box = (100.0, 100.0, 100.0 + width, 100.0 + width * 0.67)
        return [DetectedEntity("raw", "town_center", box, (0.0, 0.0), 0.9)]


def test_game_source_shrinks_frames_once_objects_measure_too_large() -> None:
    source = GameSource(detector=_Detector(1.5))  # type: ignore[arg-type]
    pixels = np.full((1898, 3024, 3), 40, dtype=np.uint8)
    jpeg = Image.fromarray(pixels)
    buffer = io.BytesIO()
    jpeg.save(buffer, format="JPEG")

    async def run() -> tuple[EntitySnapshot, ...]:
        found: tuple[EntitySnapshot, ...] = ()
        for _ in range(MIN_SAMPLES + 1):
            found = await source._detect(buffer.getvalue(), pixels)
        return found

    (last,) = asyncio.run(run())
    detector = source._detector
    assert isinstance(detector, _Detector)
    assert detector.sizes[:MIN_SAMPLES] == [1.0] * MIN_SAMPLES
    assert detector.sizes[-1] == pytest.approx(0.65, abs=0.01)
    # The detector saw a matched Town Center; the agent sees it at frame size.
    assert last.bbox is not None
    assert (last.bbox[2] - last.bbox[0]) / 3024 == pytest.approx(REFERENCE_TC_WIDTH * 1.5, rel=0.02)
