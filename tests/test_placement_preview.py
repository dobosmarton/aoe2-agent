"""The game's placement preview turns red where a building can't go."""

from __future__ import annotations

import numpy as np
from gameplay_agent.placement_preview import (
    PreviewKind,
    PreviewReading,
    PreviewRegion,
    changed_share,
    preview_refused,
    read_preview,
)

_SIZE = (400, 300)
_SPOT = (200, 200)
_REGION = PreviewRegion.around(_SPOT, 50.0, 25.0, _SIZE)


def _frame(*, red_at: PreviewRegion | None = None) -> np.ndarray:
    pixels = np.full((_SIZE[1], _SIZE[0], 3), (90, 120, 60), dtype=np.uint8)  # Grass.
    if red_at is not None:
        pixels[red_at.y0 : red_at.y1, red_at.x0 : red_at.x1] = (210, 40, 40)
    return pixels


def test_a_preview_in_normal_colours_is_accepted() -> None:
    assert not preview_refused(_frame(), _frame(), _REGION)


def test_a_red_preview_is_refused() -> None:
    assert preview_refused(_frame(), _frame(red_at=_REGION), _REGION)


def test_red_that_was_there_before_the_preview_is_not_a_refusal() -> None:
    already_red = _frame(red_at=_REGION)
    assert not preview_refused(already_red, already_red, _REGION)


def test_the_region_covers_the_sprite_above_the_footprint() -> None:
    assert PreviewRegion(x0=150, y0=100, x1=250, y1=225) == _REGION


def test_a_preview_changes_its_whole_region() -> None:
    assert changed_share(_frame(), _frame(red_at=_REGION), _REGION) == 1.0


def test_an_unchanged_region_has_no_changed_share() -> None:
    assert changed_share(_frame(), _frame(), _REGION) == 0.0


def test_a_reading_keeps_the_numbers_behind_its_verdict() -> None:
    reading = read_preview(_frame(), _frame(red_at=_REGION), _REGION)
    assert reading == PreviewReading(red_before=0.0, red=1.0, changed=1.0)


_FARM = PreviewRegion.around(_SPOT, 50.0, 25.0, _SIZE, PreviewKind.OUTLINE)


def _outline(color: tuple[int, int, int]) -> np.ndarray:
    """Grass with a one-pixel outline along the top of a farm's footprint."""
    pixels = _frame()
    pixels[_FARM.y0, _FARM.x0 : _FARM.x1] = color
    return pixels


def test_a_farm_is_read_over_its_footprint_alone() -> None:
    assert PreviewRegion(142, 171, 257, 228, PreviewKind.OUTLINE) == _FARM


def test_a_thin_red_outline_refuses_a_farm() -> None:
    assert read_preview(_frame(), _outline((210, 40, 40)), _FARM).refused


def test_a_light_outline_accepts_a_farm() -> None:
    assert read_preview(_frame(), _outline((235, 235, 235)), _FARM).accepted


def test_a_sprite_with_nothing_drawn_is_not_accepted() -> None:
    assert not read_preview(_frame(), _frame(), _REGION).accepted
