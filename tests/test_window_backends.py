"""The Mac window backend and capture, against fake Quartz window lists and images."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("Quartz")

from gameplay_agent import screen_macos
from gameplay_agent import window_macos as wm
from gameplay_agent.capture import Frame
from gameplay_agent.window import GameGeometry

from tests.window_fakes import MAC_GAME

GAME_PID = 7


def _row(pid: int, layer: int, bounds: tuple[float, float, float, float], number: int) -> dict:
    x, y, width, height = bounds
    return {
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowNumber": number,
        "kCGWindowBounds": {"X": x, "Y": y, "Width": width, "Height": height},
    }


class _App:
    def __init__(self, on_activate: object = None) -> None:
        self.activations = 0
        self._on_activate = on_activate

    def processIdentifier(self) -> int:
        return GAME_PID

    def bundleIdentifier(self) -> str:
        return "com.example.game"

    def activateWithOptions_(self, options: int) -> bool:
        del options
        self.activations += 1
        if callable(self._on_activate):
            self._on_activate()
        return True


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 0.0, 1512.0, 982.0), (0.0, 33.0, 1512.0, 949.0)],
    ids=["window under the menu bar", "window below the menu bar"],
)
def test_the_game_area_leaves_out_the_band_either_way(
    bounds: tuple[float, float, float, float],
) -> None:
    assert wm.geometry_from(bounds, 33.0, 2.0, 42) == MAC_GAME


def test_a_window_entirely_under_the_menu_bar_has_no_height_left() -> None:
    assert wm.game_area_points((0.0, 0.0, 100.0, 20.0), 33.0) == (0.0, 20.0, 100.0, 0.0)


def test_the_game_window_is_its_largest_normal_window() -> None:
    rows = [
        _row(GAME_PID, 0, (0, 33, 200, 100), 1),
        _row(GAME_PID, 0, (0, 33, 1512, 949), 2),
        _row(GAME_PID, 25, (0, 0, 1512, 982), 3),  # An overlay layer, even though larger.
        _row(9, 0, (0, 0, 1512, 982), 4),  # Another app.
    ]
    picked = wm.pick_game_window(rows, GAME_PID)
    assert picked is not None
    assert picked["kCGWindowNumber"] == 2
    assert wm.pick_game_window([_row(9, 0, (0, 0, 10, 10), 1)], GAME_PID) is None


def test_the_front_window_is_the_first_normal_one() -> None:
    rows = [_row(1, 25, (0, 0, 1512, 33), 1), _row(GAME_PID, 0, (0, 33, 1512, 949), 2)]
    assert wm.front_window_pid(rows) == GAME_PID
    assert wm.front_window_pid([_row(1, 25, (0, 0, 1512, 33), 1)]) is None


@pytest.fixture
def screen_state(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake desktop: which app is in front, and the game's window."""
    state = SimpleNamespace(front_pid=GAME_PID, app=_App(), opened=[], game_x=0.0)

    def rows(options: int) -> list[dict]:
        del options
        game_row = _row(GAME_PID, 0, (state.game_x, 33, 1512, 949), 42)
        if state.front_pid == GAME_PID:
            return [game_row]
        return [_row(state.front_pid, 0, (0, 33, 1512, 949), 99), game_row]

    monkeypatch.setattr(wm, "find_game_app", lambda: state.app)
    monkeypatch.setattr(wm, "_window_rows", rows)
    monkeypatch.setattr(wm, "menu_bar_points", lambda: 33.0)
    monkeypatch.setattr(wm, "display_scale", lambda: 2.0)
    monkeypatch.setattr(wm, "screen_span_points", lambda: (0.0, 1512.0))
    monkeypatch.setattr(wm, "FOCUS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(wm, "OPEN_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(wm, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(wm.subprocess, "run", lambda args, **_: state.opened.append(args))
    return state


def test_the_backend_finds_the_game_area_in_frame_pixels(screen_state: SimpleNamespace) -> None:
    backend = wm.MacBackend()
    assert backend.geometry() == MAC_GAME
    assert backend.input_scale() == 2.0
    assert backend.is_running()


def test_the_game_is_in_front_only_when_its_window_leads_the_list(
    screen_state: SimpleNamespace,
) -> None:
    backend = wm.MacBackend()
    assert backend.in_front()
    screen_state.front_pid = 1234  # Terminal, say.
    assert not backend.in_front()


def test_a_game_still_sliding_in_is_not_in_front(screen_state: SimpleNamespace) -> None:
    screen_state.game_x = 1454.0  # Read mid-slide on the probe's first run.
    assert not wm.MacBackend().in_front()
    screen_state.game_x = 0.0
    assert wm.MacBackend().in_front()


@pytest.mark.parametrize(
    ("x", "inside"), [(0.0, True), (-1.0, False), (1.0, False), (1454.0, False)]
)
def test_only_a_window_within_the_screens_counts_as_on_screen(x: float, inside: bool) -> None:
    assert wm.wholly_on_screen((x, 33.0, 1512.0, 949.0), (0.0, 1512.0)) is inside


def test_no_game_means_no_window_and_not_in_front(
    screen_state: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wm, "find_game_app", lambda: None)
    backend = wm.MacBackend()
    assert backend.geometry() is None
    assert not backend.in_front()
    assert not backend.focus(3)


def test_focus_activates_the_game_when_another_app_is_in_front(
    screen_state: SimpleNamespace,
) -> None:
    screen_state.front_pid = 1234
    screen_state.app = _App(on_activate=lambda: setattr(screen_state, "front_pid", GAME_PID))
    assert wm.MacBackend().focus(3)
    assert screen_state.app.activations == 1
    assert screen_state.opened == []


def test_focus_does_nothing_when_the_game_is_already_in_front(
    screen_state: SimpleNamespace,
) -> None:
    assert wm.MacBackend().focus(3)
    assert screen_state.app.activations == 0


def test_refused_activation_falls_back_to_launch_services(
    screen_state: SimpleNamespace,
) -> None:
    screen_state.front_pid = 1234
    assert not wm.MacBackend().focus(2)
    assert screen_state.app.activations == 2
    assert screen_state.opened == [["open", "-b", "com.example.game"]]


def test_bgra_rows_become_rgb_without_row_padding() -> None:
    width, height, row_bytes = 2, 2, 16  # Rows padded to 4 pixels.
    raw = bytearray(row_bytes * height)
    raw[0:4] = bytes((10, 20, 30, 255))  # Blue, green, red, alpha.
    raw[row_bytes + 4 : row_bytes + 8] = bytes((1, 2, 3, 255))
    rgb = screen_macos.bgra_to_rgb(bytes(raw), width, height, row_bytes)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]
    assert rgb[1, 1].tolist() == [3, 2, 1]


def _fake_quartz(width: int, height: int, requests: list[tuple[object, ...]]) -> SimpleNamespace:
    row_bytes = width * 4
    return SimpleNamespace(
        kCGWindowListOptionIncludingWindow=8,
        CGRectMake=lambda *area: area,
        CGWindowListCreateImage=lambda *args: requests.append(args) or "image",
        CGImageGetWidth=lambda _: width,
        CGImageGetHeight=lambda _: height,
        CGImageGetBytesPerRow=lambda _: row_bytes,
        CGImageGetDataProvider=lambda _: "provider",
        CGDataProviderCopyData=lambda _: bytes(row_bytes * height),
    )


def test_capture_asks_for_the_game_area_in_points(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[object, ...]] = []
    geometry = GameGeometry(0, 66, 8, 6, 2.0, 42)
    monkeypatch.setattr(screen_macos, "Quartz", _fake_quartz(8, 6, requests))
    frame = screen_macos.grab_game_area(geometry)
    assert frame.shape == (6, 8, 3)
    assert frame.dtype == np.uint8
    area, option, window_id, _ = requests[0]
    assert area == (0.0, 33.0, 4.0, 3.0)
    assert (option, window_id) == (8, 42)


def test_capture_refuses_a_frame_of_the_wrong_size(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = GameGeometry(0, 66, 8, 6, 2.0, 42)
    monkeypatch.setattr(screen_macos, "Quartz", _fake_quartz(4, 3, []))  # Half size.
    with pytest.raises(RuntimeError, match="captured 4x3"):
        screen_macos.grab_game_area(geometry)


def test_capture_needs_the_window_number() -> None:
    with pytest.raises(ValueError, match="window number"):
        screen_macos.grab_game_area(GameGeometry(0, 0, 8, 6, 2.0))


def test_bgra_rows_above_the_game_area_and_spare_bytes_are_dropped() -> None:
    width, height, row_bytes = 1, 3, 4
    raw = bytes((0, 0, 9, 255, 0, 0, 8, 255, 0, 0, 7, 255)) + bytes(16)  # Spare bytes.
    rgb = screen_macos.bgra_to_rgb(raw, width, height, row_bytes, top=1)
    assert rgb[:, 0, 0].tolist() == [8, 7]


def test_window_grabber_stamps_before_the_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(screen_macos, "Quartz", _fake_quartz(8, 6, []))
    frame = screen_macos.WindowGrabber().grab(GameGeometry(0, 66, 8, 6, 2.0, 42), 1)
    assert (frame.width, frame.height, len(frame.rgb)) == (8, 6, 8 * 6 * 3)


COMPLETE = screen_macos.ScreenCaptureKit.SCFrameStatusComplete
IDLE = screen_macos.ScreenCaptureKit.SCFrameStatusIdle
BLANK = 2


def test_idle_notices_keep_the_last_image_current() -> None:
    latest = screen_macos.LatestFrame()
    latest.offer("first", COMPLETE, 1.0)
    latest.offer(None, IDLE, 2.0)
    assert latest.take(0) == ("first", 2.0)


def test_blank_frames_and_idle_before_any_image_are_ignored() -> None:
    latest = screen_macos.LatestFrame()
    latest.offer(None, IDLE, 1.0)
    with pytest.raises(RuntimeError, match="no frame"):
        latest.take(0)
    latest.offer("image", COMPLETE, 2.0)
    latest.offer("blank", BLANK, 3.0)
    assert latest.take(0) == ("image", 2.0)


class _FakeStream:
    def __init__(self, geometry: GameGeometry, opened: list[GameGeometry]) -> None:
        opened.append(geometry)
        self.top = 66
        self.stopped = False

    def take(self) -> tuple[object, float]:
        return "buffer", 7.5

    def stop(self) -> None:
        self.stopped = True


def _stream_grabber(opened: list[GameGeometry], rows: list[int]) -> screen_macos.StreamGrabber:
    streams: list[_FakeStream] = []

    def open_stream(geometry: GameGeometry) -> _FakeStream:
        streams.append(_FakeStream(geometry, opened))
        return streams[-1]

    def to_rgb(buffer: object, top: int) -> np.ndarray:
        assert buffer == "buffer"
        rows.append(top)
        geometry = opened[-1]
        return np.zeros((geometry.height, geometry.width, 3), dtype=np.uint8)

    grabber = screen_macos.StreamGrabber(open_stream, to_rgb, clock=lambda: 7.6)
    grabber.streams = streams  # type: ignore[attr-defined]
    return grabber


def test_the_stream_starts_once_and_gives_the_game_area() -> None:
    opened: list[GameGeometry] = []
    rows: list[int] = []
    grabber = _stream_grabber(opened, rows)
    first = grabber.grab(MAC_GAME, 1)
    grabber.grab(MAC_GAME, 1)
    assert opened == [MAC_GAME]
    assert rows == [66, 66]  # The band above the game area is cut.
    assert (first.width, first.height, first.captured_at) == (3024, 1898, 7.5)


def test_the_stream_restarts_when_the_window_changes_and_stops_on_close() -> None:
    opened: list[GameGeometry] = []
    grabber = _stream_grabber(opened, [])
    grabber.grab(MAC_GAME, 1)
    moved = GameGeometry(0, 66, 3024, 1898, 2.0, 43)
    grabber.grab(moved, 1)
    assert opened == [MAC_GAME, moved]
    first, second = grabber.streams  # type: ignore[attr-defined]
    assert first.stopped and not second.stopped
    grabber.close()
    assert second.stopped


def test_the_stream_refuses_a_frame_of_the_wrong_size() -> None:
    def half_size(_buffer: object, _top: int) -> np.ndarray:
        return np.zeros((949, 1512, 3), dtype=np.uint8)

    grabber = screen_macos.StreamGrabber(lambda g: _FakeStream(g, []), half_size, clock=lambda: 7.6)
    with pytest.raises(RuntimeError, match="stream gave 1512x949"):
        grabber.grab(MAC_GAME, 1)


def test_the_stream_needs_the_window_number() -> None:
    with pytest.raises(ValueError, match="window number"):
        screen_macos.StreamGrabber().grab(GameGeometry(0, 0, 8, 6, 2.0), 1)


def test_an_old_stream_frame_is_replaced_by_a_still_capture() -> None:
    stills: list[GameGeometry] = []

    def still(area: GameGeometry) -> Frame:
        stills.append(area)
        return Frame(b"", area.width, area.height, 9.0)

    def no_pixels(_buffer: object, _top: int) -> np.ndarray:
        raise AssertionError("an old stream frame must not be used")

    now = [7.6]  # The fake stream's frame arrived at 7.5.
    grabber = screen_macos.StreamGrabber(
        lambda g: _FakeStream(g, []), no_pixels, still=still, clock=lambda: now[0]
    )
    now[0] = 7.5 + screen_macos.STREAM_MAX_AGE_SECONDS + 0.01  # Paused: no new frames.
    frame = grabber.grab(MAC_GAME, 1)
    assert stills == [MAC_GAME]
    assert frame.captured_at == 9.0  # The still's own time, not the old frame's.
