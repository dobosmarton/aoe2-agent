"""The Mac preflight's frame checks and its cursor round trip."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gameplay_agent import preflight
from gameplay_agent.window import GameGeometry

from tests.window_fakes import MAC_GAME


class _Pointer:
    """pyautogui on a 1512-point-wide screen. The cursor lands `drift` points away
    from where it was sent, or stays put if `moves` is false."""

    def __init__(self, drift: tuple[int, int] = (0, 0), *, moves: bool = True) -> None:
        self.at = (0, 0)
        self.drift = drift
        self.moves = moves

    def size(self) -> SimpleNamespace:
        return SimpleNamespace(width=1512, height=982)

    def moveTo(self, x: int, y: int) -> None:
        if self.moves:
            self.at = (x + self.drift[0], y + self.drift[1])

    def position(self) -> SimpleNamespace:
        return SimpleNamespace(x=self.at[0], y=self.at[1])


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "_CURSOR_SETTLE_SECONDS", 0.0)


def test_the_round_trip_passes_when_the_cursor_lands_mid_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyautogui", _Pointer())
    check = preflight._cursor_round_trip(MAC_GAME, input_allowed=True, frame_width=3024)
    assert check.passed, check.detail


def test_the_round_trip_fails_when_input_is_off_by_a_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyautogui", _Pointer(drift=(0, 1)))  # Two frame pixels.
    assert not preflight._cursor_round_trip(MAC_GAME, True, frame_width=3024).passed


def test_the_round_trip_fails_when_the_cursor_never_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyautogui", _Pointer(moves=False))
    assert not preflight._cursor_round_trip(MAC_GAME, True, frame_width=3024).passed


def test_the_round_trip_catches_unscaled_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyautogui", _Pointer())
    unscaled = GameGeometry(0, 66, 3024, 1898, 1.0, 42)  # The VM's arithmetic on a Mac.
    check = preflight._cursor_round_trip(unscaled, True, frame_width=3024)
    assert not check.passed
    assert "scale 2" in check.detail


def test_a_half_size_capture_fails_the_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyautogui", _Pointer())
    assert not preflight._cursor_round_trip(MAC_GAME, True, frame_width=1512).passed


def test_the_round_trip_needs_the_window_and_the_permission() -> None:
    assert not preflight._cursor_round_trip(None, True, frame_width=3024).passed
    skipped = preflight._cursor_round_trip(MAC_GAME, False, frame_width=3024)
    assert not skipped.passed
    assert "skipped" in skipped.detail


def test_a_black_band_fails_and_a_game_frame_passes() -> None:
    frame = np.full((40, 8, 3), 120, dtype=np.uint8)
    assert preflight._band_check(frame).passed
    frame[:10] = 0
    assert not preflight._band_check(frame).passed


def test_warnings_do_not_block() -> None:
    warning = preflight.PreflightCheck("keyboard_layout", False, "Hungarian", blocking=False)
    assert warning.blocking is False
    assert preflight.PreflightCheck("window", False, "missing").blocking
