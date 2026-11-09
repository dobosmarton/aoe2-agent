"""A menu or popup dims the game's fixed HUD artwork, and that is what is checked."""

from __future__ import annotations

import numpy as np
import pytest
from gameplay_agent.game_screen import hud_brightness, popup_open


def _frame(bar: int, panel: int, *, world: int = 120) -> np.ndarray:
    """A 3024x1898 frame with the resource bar, bottom panel and world at set levels."""
    frame = np.full((1898, 3024, 3), world, dtype=np.uint8)
    frame[:80, :1288] = bar
    frame[1640:, 429:1400] = panel
    return frame


def test_the_hud_regions_are_measured() -> None:
    bar, panel = hud_brightness(_frame(66, 150))
    assert (bar, panel) == pytest.approx((66, 150), abs=1)


@pytest.mark.parametrize("world", [0, 120, 255])  # Night, grass, snow: the world doesn't count.
def test_a_normal_hud_is_not_a_popup(world: int) -> None:
    assert not popup_open(_frame(66, 150, world=world))


def test_the_main_menu_dims_both_regions() -> None:
    assert popup_open(_frame(13, 31))  # Measured with the Main Menu open.


def test_one_dark_region_alone_is_not_a_popup() -> None:
    assert not popup_open(_frame(13, 150))
    assert not popup_open(_frame(66, 31))
