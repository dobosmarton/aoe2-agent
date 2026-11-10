"""Every platform captures through one interface, and Windows keeps mss."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from gameplay_agent import capture, screen
from gameplay_agent.capture import FallbackGrabber, MssGrabber, make_grabber, resolve_method
from gameplay_agent.config import config

from tests.window_fakes import VM_GAME, FakeGrabber


@pytest.mark.parametrize(
    ("method", "platform", "resolved"),
    [
        ("auto", "windows", "mss"),
        ("auto", "macos", "stream"),
        ("mss", "windows", "mss"),
        ("stream", "macos", "stream"),
        ("window", "macos", "window"),
    ],
)
def test_each_platform_resolves_its_own_method(method: str, platform: str, resolved: str) -> None:
    assert resolve_method(method, platform) == resolved  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "platform"), [("stream", "windows"), ("window", "windows"), ("mss", "macos")]
)
def test_a_method_the_platform_lacks_is_refused(method: str, platform: str) -> None:
    with pytest.raises(ValueError, match=f"{method!r} doesn't work on {platform}"):
        resolve_method(method, platform)  # type: ignore[arg-type]


def test_windows_never_imports_the_mac_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gameplay_agent.screen_macos", None)  # Import fails.
    assert isinstance(make_grabber("auto", "windows"), MssGrabber)


class _Shot:
    size = (2, 1)
    bgra = bytes((10, 20, 30, 0, 1, 2, 3, 0))


class _FakeMss:
    def __init__(self, regions: list[object]) -> None:
        self.regions = regions
        self.monitors = [None, {"left": 0, "top": 0, "width": 2, "height": 1}]

    def __enter__(self) -> _FakeMss:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def grab(self, region: object) -> _Shot:
        self.regions.append(region)
        return _Shot()


def test_mss_grabs_the_window_region_and_falls_back_to_the_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regions: list[object] = []
    monkeypatch.setattr(capture, "mss", SimpleNamespace(MSS=lambda: _FakeMss(regions)))
    grabber = MssGrabber()
    frame = grabber.grab(VM_GAME, 1)
    grabber.grab(None, 1)
    assert regions == [
        {"left": 200, "top": 100, "width": 3024, "height": 1672},
        {"left": 0, "top": 0, "width": 2, "height": 1},
    ]
    assert (frame.width, frame.height) == (2, 1)
    assert frame.rgb == bytes((30, 20, 10, 3, 2, 1))  # BGRX became RGB.


class _Broken:
    def __init__(self) -> None:
        self.grabs = 0
        self.closed = False

    def grab(self, geometry: object, monitor: int, *, after: float | None = None) -> object:
        del geometry, monitor, after
        self.grabs += 1
        raise RuntimeError("ScreenCaptureKit refused")

    def close(self) -> None:
        self.closed = True


def test_a_failing_primary_is_dropped_for_the_rest_of_the_run() -> None:
    primary, fallback = _Broken(), FakeGrabber()
    grabber = FallbackGrabber(primary, fallback)  # type: ignore[arg-type]
    grabber.grab(VM_GAME, 1)
    grabber.grab(VM_GAME, 1)
    assert primary.grabs == 1
    assert primary.closed
    assert fallback.grabs == [VM_GAME, VM_GAME]


def test_the_run_builds_its_grabber_once_and_close_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, str]] = []
    fake = FakeGrabber()

    def make(method: str, platform: str) -> FakeGrabber:
        built.append((method, platform))
        return fake

    monkeypatch.setattr(screen, "make_grabber", make)
    monkeypatch.setattr(screen, "_grabber", None)
    monkeypatch.setattr(config, "capture_method", "auto")
    assert screen.frame_grabber() is screen.frame_grabber()
    assert built == [("auto", "none")]  # The tests' null window backend.
    screen.close_capture()
    assert fake.closed
    assert screen._grabber is None
