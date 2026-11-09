"""Frame pixels and input units line up, and a Mac capture is the game area or nothing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from gameplay_agent import screen, window
from gameplay_agent.window import GameGeometry, NullBackend, to_input_point

from tests.window_fakes import MAC_GAME, FakeGrabber, FakeWindowBackend


@pytest.mark.parametrize("offset", [(0, 0), (200, 100), (-8, 31)])
@pytest.mark.parametrize("point", [(0, 0), (1512, 836), (3023, 1671)])
def test_at_scale_one_input_is_the_frame_point_plus_the_window_origin(
    offset: tuple[int, int], point: tuple[int, int]
) -> None:
    assert to_input_point(*point, *offset, 1.0) == (point[0] + offset[0], point[1] + offset[1])


def test_a_retina_frame_pixel_is_half_an_input_point_below_the_band() -> None:
    assert MAC_GAME.to_input(0, 0) == (0, 33)
    assert MAC_GAME.to_input(1000, 800) == (500, 433)
    assert MAC_GAME.to_frame(500, 433) == (1000, 800)


def test_an_odd_frame_pixel_comes_back_within_one_pixel() -> None:
    point = (1511, 949)
    back = MAC_GAME.to_frame(*MAC_GAME.to_input(*point))
    assert max(abs(back[0] - point[0]), abs(back[1] - point[1])) <= 1


def test_the_backend_supplies_the_rect_scale_and_platform() -> None:
    with window.use_backend(FakeWindowBackend()):
        assert window.get_game_window_rect() == (0, 66, 3024, 1898)
        assert window.get_input_scale() == 2.0
        assert window.game_platform() == "macos"
    assert isinstance(window._current(), NullBackend)
    assert window.get_game_window_rect() is None
    assert window.get_input_scale() == 1.0


def test_a_windows_geometry_keeps_the_old_rect_tuple() -> None:
    geometry = GameGeometry(200, 100, 3024, 1672)
    assert geometry.rect == (200, 100, 3024, 1672)
    assert geometry.scale == 1.0
    assert geometry.window_id is None


def test_a_mac_capture_is_the_game_area_with_the_cursor_in_frame_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer = SimpleNamespace(position=lambda: SimpleNamespace(x=500, y=433))
    monkeypatch.setitem(sys.modules, "pyautogui", pointer)
    with window.use_backend(FakeWindowBackend()), screen.use_grabber(FakeGrabber()) as grabber:
        capture = screen.capture_native()
    assert grabber.grabs == [MAC_GAME]
    assert (capture.width, capture.height) == (3024, 1898)
    assert capture.window_rect == (0, 66, 3024, 1898)
    assert capture.cursor == (1000, 800)
    assert capture.captured_at == grabber.at  # When the screen looked like this.


def test_a_windows_capture_without_the_window_grabs_the_monitor() -> None:
    backend = FakeWindowBackend(platform="windows", game=None)
    with window.use_backend(backend), screen.use_grabber(FakeGrabber()) as grabber:
        capture = screen.capture_native()
    assert grabber.grabs == [None]
    assert capture.window_rect is None


def test_a_mac_capture_without_the_window_raises_instead_of_grabbing_the_display() -> None:
    with (
        window.use_backend(FakeWindowBackend(game=None)),
        pytest.raises(RuntimeError, match="wasn't found"),
    ):
        screen.capture_native()
