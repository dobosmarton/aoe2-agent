"""The Mac probe's pure helpers. The probe itself needs the game and the permissions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Quartz")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import macos_probe


def _window(pid: int, layer: int, width: float, height: float, number: int) -> dict[str, object]:
    return {
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowNumber": number,
        "kCGWindowBounds": {"X": 0.0, "Y": 33.0, "Width": width, "Height": height},
    }


def test_the_game_window_is_its_largest_normal_window() -> None:
    rows = [
        _window(7, 0, 200.0, 100.0, 1),
        _window(7, 0, 1512.0, 949.0, 2),
        _window(7, 25, 1512.0, 982.0, 3),  # An overlay layer, even though larger.
        _window(9, 0, 1512.0, 982.0, 4),  # Another app.
    ]
    picked = macos_probe.pick_game_window(rows, 7)
    assert picked is not None
    assert picked["kCGWindowNumber"] == 2


def test_no_window_when_the_game_has_none() -> None:
    assert macos_probe.pick_game_window([_window(9, 0, 10.0, 10.0, 1)], 7) is None


def test_bgra_rows_become_rgb_without_row_padding() -> None:
    width, height, row_bytes = 2, 2, 16  # Rows padded to 4 pixels.
    raw = bytearray(row_bytes * height)
    raw[0:4] = bytes((10, 20, 30, 255))  # Blue, green, red, alpha.
    raw[row_bytes + 4 : row_bytes + 8] = bytes((1, 2, 3, 255))
    rgb = macos_probe.bgra_to_rgb(bytes(raw), width, height, row_bytes)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]
    assert rgb[1, 1].tolist() == [3, 2, 1]


def test_a_black_band_is_seen_only_at_the_top() -> None:
    frame = np.full((40, 8, 3), 120, dtype=np.uint8)
    assert not macos_probe.top_is_black(frame)
    frame[: macos_probe.BAND_ROWS] = 0
    assert macos_probe.top_is_black(frame)


SCREEN_WIDTH = 1512.0
RESTING = {"X": 0.0, "Y": 33.0, "Width": 1512.0, "Height": 949.0}


def test_a_window_still_sliding_in_is_not_at_rest() -> None:
    sliding = {**RESTING, "X": 1454.0}
    assert not macos_probe.at_rest(sliding, sliding, SCREEN_WIDTH)
    assert not macos_probe.at_rest(sliding, RESTING, SCREEN_WIDTH)


def test_a_window_is_at_rest_once_it_stays_on_the_screen() -> None:
    assert macos_probe.at_rest(dict(RESTING), dict(RESTING), SCREEN_WIDTH)
    assert not macos_probe.at_rest(None, dict(RESTING), SCREEN_WIDTH)
    assert not macos_probe.at_rest(dict(RESTING), None, SCREEN_WIDTH)


def test_spare_bytes_after_the_last_row_are_ignored() -> None:
    width, height, row_bytes = 2, 2, 8
    raw = bytes(range(row_bytes * height)) + bytes(5)
    rgb = macos_probe.bgra_to_rgb(raw, width, height, row_bytes)
    assert rgb.shape == (2, 2, 3)
    assert rgb[1, 1].tolist() == [14, 13, 12]
