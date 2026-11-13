"""Before clicking, the agent hovers candidate spots and lets the preview judge them."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import gameplay_agent.executor as ex
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_SPOTS = [(1000, 900), (1400, 900), (1800, 900)]
_ACTION: dict[str, object] = {"auto_placement": True, "building_key": "q", "menu": "q"}


class _Cursor:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []

    def moveTo(self, x: int, y: int) -> None:
        self.moves.append((x, y))


def _capture_red_at(cursor: _Cursor, red: set[tuple[int, int]]) -> ex.PreviewCapture:
    width, height = ex._window_size()

    async def capture(_after: float) -> np.ndarray:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = cursor.moves[-1]
        if (x, y) in red:
            pixels[y - 100 : y + 20, x - 60 : x + 60] = (210, 40, 40)
        return pixels

    return capture


@pytest.fixture
def cursor(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Cursor]:
    fake = _Cursor()
    monkeypatch.setattr(ex, "pyautogui", fake)
    monkeypatch.setattr(ex, "_spatial_rejection", lambda _action: None)
    monkeypatch.setattr(ex, "placement_candidates", lambda _key, *, menu="q": list(_SPOTS))
    ex._window_offset, ex._input_scale = (0, 0), 1.0
    yield fake


def _place(capture: ex.PreviewCapture) -> tuple[str, tuple[int, int] | None]:
    return asyncio.run(ex._previewed_placement(_ACTION, capture))


def test_the_first_spot_the_preview_accepts_is_chosen(cursor: _Cursor) -> None:
    assert _place(_capture_red_at(cursor, {_SPOTS[0]})) == ("", _SPOTS[1])


def test_every_spot_red_means_no_placement(cursor: _Cursor) -> None:
    detail, point = _place(_capture_red_at(cursor, set(_SPOTS)))
    assert (point, "preview was red at 3 spots" in detail) == (None, True)


def test_the_cursor_rests_over_the_command_panel_before_hovering(cursor: _Cursor) -> None:
    _place(_capture_red_at(cursor, set()))
    _width, height = ex._window_size()
    assert cursor.moves[0][1] > height - ex.UI_MARGIN_BOTTOM
