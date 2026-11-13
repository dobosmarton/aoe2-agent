"""An age-up's progress bar shows the next age's name; it is not the current age."""

from __future__ import annotations

import numpy as np
from gameplay_agent.resource_ocr import FieldBox, age_up_in_progress

_AGE_BOX = FieldBox(1000, 18, 1360, 74)
_BAR_RED = (200, 30, 30)


def _hud(*, bar_from: int | None = None, bar_to: int = 1010) -> np.ndarray:
    rgb = np.full((120, 1500, 3), (40, 30, 20), dtype=np.uint8)  # The dark age field.
    if bar_from is not None:
        rgb[26:61, bar_from:bar_to] = _BAR_RED
    return rgb


def test_a_dark_field_is_not_an_age_up() -> None:
    assert not age_up_in_progress(_hud(), _AGE_BOX)


def test_a_bar_starting_left_of_the_field_is_an_age_up() -> None:
    assert age_up_in_progress(_hud(bar_from=988), _AGE_BOX)


def test_red_inside_the_field_alone_is_not_an_age_up() -> None:
    assert not age_up_in_progress(_hud(bar_from=1100, bar_to=1200), _AGE_BOX)
