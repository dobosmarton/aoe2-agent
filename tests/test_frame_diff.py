"""Camera-pan estimation, and the detection ladder built on it.

Run 2026_08_22_1 paid ~2 s for each of 112 detections whose view had merely
panned. A translated cache is still true, and costs nothing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from detection.inference.frame_diff import FrameChange, FrameDiffer
from gameplay_agent.detection_phase import (
    STATIC_CLASSES,
    _pan_translation,
    _translated_static,
)
from PIL import Image

if TYPE_CHECKING:
    from gameplay_agent.entity_snapshot import EntitySnapshot

_SIZE = (640, 360)


def _noise_frame(seed: int) -> Image.Image:
    """A textured frame — phase correlation needs detail to lock onto."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*_SIZE[::-1], 3), dtype=np.uint8))


def _jpg(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _panned(img: Image.Image, dx: int, dy: int) -> Image.Image:
    return Image.fromarray(np.roll(np.array(img), (dy, dx), axis=(0, 1)))


@pytest.mark.parametrize(("dx", "dy"), [(80, 0), (0, 40), (-120, 60)])
def test_a_pan_is_measured_in_screen_pixels(dx: int, dy: int) -> None:
    base = _noise_frame(1)
    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(base))
    change = differ.compare(_jpg(_panned(base, dx, dy)))
    assert change.response > 0.7
    assert change.shift == pytest.approx((dx, dy), abs=2)


def test_unrelated_frames_report_low_confidence() -> None:
    """The caller refuses to translate on a weak response, so it must be low
    when the view did more than pan."""
    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(_noise_frame(1)))
    assert differ.compare(_jpg(_noise_frame(2))).response < 0.7


def test_an_unchanged_frame_reports_no_change() -> None:
    frame = _jpg(_noise_frame(1))
    differ = FrameDiffer(threshold=0.03)
    differ.compare(frame)
    assert differ.compare(frame).changed is False


def test_the_first_frame_has_nothing_to_compare_against() -> None:
    assert FrameDiffer(threshold=0.03).compare(_jpg(_noise_frame(1))).changed is True


def test_a_static_entity_moves_with_the_content() -> None:
    """The shift is how far the CONTENT moved, so it adds."""
    moved = _translated_static([{"class": "tree", "center": (100, 100)}], (120.0, 60.0))
    assert moved[0]["center"] == (220, 160)


def test_a_moving_entity_is_dropped_rather_than_translated() -> None:
    """A villager that walked is worse than one the caller is not told about."""
    entities = [{"class": "villager", "center": (1, 1)}, {"class": "sheep", "center": (2, 2)}]
    assert _translated_static(entities, (10.0, 10.0)) == []


def test_herd_animals_are_not_static() -> None:
    """CLASSES_BY_KIND['food'] mixes berry bushes with sheep, boar and deer."""
    assert "berry_bush" in STATIC_CLASSES
    assert not {"sheep", "boar", "deer"} & STATIC_CLASSES


# ---------------------------------------------------------------------------
# Accuracy on a real game frame
# ---------------------------------------------------------------------------
# Noise frames prove the maths; a real screenshot proves it survives JPEG
# artefacts, the isometric terrain and the HUD. `logs/` is gitignored, so this
# skips wherever a run is not present.

_MAX_PAN_ERROR_PX = 4


def _a_real_frame() -> Image.Image | None:
    runs = sorted(Path("logs").glob("*/images/*.jpg"))
    return Image.open(runs[len(runs) // 2]) if runs else None


@pytest.mark.parametrize(("dx", "dy"), [(150, 0), (0, -90), (-240, 130)])
def test_a_real_frame_pans_within_a_few_pixels(dx: int, dy: int) -> None:
    """A click needs the target within its footprint, not to the pixel."""
    frame = _a_real_frame()
    if frame is None:
        pytest.skip("no recorded run under logs/")
    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(frame))
    change = differ.compare(_jpg(_panned(frame, dx, dy)))
    assert change.response > 0.7
    assert change.shift == pytest.approx((dx, dy), abs=_MAX_PAN_ERROR_PX)


def test_a_missing_opencv_is_announced_not_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing the pan loses the whole rescan cache, so it must not be silent."""
    import builtins

    from detection.inference import frame_diff

    real_import = builtins.__import__

    def no_cv2(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2":
            raise ImportError("stubbed")
        return real_import(name, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    monkeypatch.setattr(frame_diff, "_cv2_warned", False)

    base = _noise_frame(1)
    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(base))
    with caplog.at_level("WARNING"):
        change = differ.compare(_jpg(_panned(base, 80, 0)))

    assert change.response == 0.0  # so the caller falls back to detecting
    assert "opencv missing" in caplog.text


# ---------------------------------------------------------------------------
# Why a pan was refused
# ---------------------------------------------------------------------------
# Run 2026_08_22_2 translated 0 of 195 rescans and the log could not say which
# clause declined, because only the success path logged.


@pytest.mark.parametrize(
    ("change", "cached", "reason"),
    [
        (
            FrameChange(True, (5.0, 5.0), 0.2),
            [{"class": "tree", "center": (10, 10)}],
            "low_confidence",
        ),
        (FrameChange(True, (5.0, 5.0), 0.9), [], "empty_cache"),
        (
            FrameChange(True, (5.0, 5.0), 0.9),
            [{"class": "villager", "center": (1, 1)}],
            "nothing_static_cached",
        ),
    ],
    ids=["low-confidence", "empty-cache", "no-static-entities"],
)
def test_every_refusal_names_itself(change: FrameChange, cached: list[dict], reason: str) -> None:
    translated, declined = _pan_translation(change, cached)
    assert translated == []
    assert declined == reason


def test_a_confident_pan_is_served_with_no_reason() -> None:
    translated, declined = _pan_translation(
        FrameChange(True, (5.0, 5.0), 0.9), [{"class": "tree", "center": (10, 10)}]
    )
    assert translated == [{"class": "tree", "center": (15, 15)}]
    assert declined == ""


# ---------------------------------------------------------------------------
# A disabled cache stays silent
# ---------------------------------------------------------------------------
# AOE2_RESCAN_CACHE=false is the A/B condition. A refusal logged on every frame
# would bury the signal the comparison exists to collect.


def _drive_one_frame(monkeypatch: pytest.MonkeyPatch, *, cache_on: bool) -> list[object]:
    """Run one `detect_frame` over 2 unrelated frames, so the pan must decline."""
    import asyncio
    from types import SimpleNamespace

    from detection.inference.frame_diff import FrameDiffer
    from gameplay_agent import detection_phase as dp
    from gameplay_agent import executor as ex

    monkeypatch.setattr(dp.config, "rescan_cache", cache_on, raising=False)
    ex.set_detected_entities([{"class": "tree", "center": (10, 10)}])

    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(_noise_frame(1)))  # a previous frame, so a pan is measurable
    detector = SimpleNamespace(tracker=None, detect_fast=lambda _s: [])
    frame = dp.detect_frame(detector, differ, _jpg(_noise_frame(2)))  # pyright: ignore[reportArgumentType]
    return asyncio.run(frame)


def test_a_disabled_cache_logs_no_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    _drive_one_frame(monkeypatch, cache_on=False)
    assert "frame_pan_declined" not in capsys.readouterr().out


def test_an_enabled_cache_reports_why_it_declined(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two unrelated frames cannot correlate, so the refusal must say so."""
    capsys.readouterr()
    _drive_one_frame(monkeypatch, cache_on=True)
    assert "low_confidence" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# detect_frame — the ladder, cheapest rung first
# ---------------------------------------------------------------------------


class _Tracker:
    """A tracker whose confidence and prediction the test dictates."""

    def __init__(self, confidence: float, predicted: list[object] | None = None) -> None:
        self.confidence = confidence
        self.predicted = predicted or []
        self.resets = 0

    def get_confidence(self) -> float:
        return self.confidence

    def predict(self) -> list[object]:
        return self.predicted

    def reset(self) -> None:
        self.resets += 1


class _Detector:
    """Counts how often the ladder had to pay for a real detection."""

    def __init__(
        self, entities: list[object] | None = None, tracker: _Tracker | None = None
    ) -> None:
        self.tracker = tracker
        self.entities = entities or []
        self.detections = 0

    def detect_fast(self, _screenshot: bytes) -> list[object]:
        self.detections += 1
        return self.entities


def _run_ladder(
    detector: _Detector, differ: FrameDiffer | None, frame: bytes
) -> tuple[EntitySnapshot, ...]:
    import asyncio

    from gameplay_agent import detection_phase as dp

    return asyncio.run(dp.detect_frame(detector, differ, frame))  # pyright: ignore[reportArgumentType]


@pytest.fixture
def _cache():
    from gameplay_agent import executor as ex

    ex.clear_detected_entities()
    yield ex
    ex.clear_detected_entities()


def _settled_differ(frame: bytes) -> FrameDiffer:
    """A differ that has already seen `frame`, so the next compare is a real one."""
    differ = FrameDiffer(threshold=0.03)
    differ.compare(frame)
    return differ


def test_an_unchanged_frame_is_served_from_the_tracker(_cache) -> None:
    """The cheapest rung: nothing moved, so extrapolate instead of detecting."""
    frame = _jpg(_noise_frame(1))
    detector = _Detector(tracker=_Tracker(confidence=0.9, predicted=[{"class": "tree"}]))
    assert [
        entity.class_name for entity in _run_ladder(detector, _settled_differ(frame), frame)
    ] == ["tree"]


def test_an_unchanged_frame_costs_no_detection(_cache) -> None:
    frame = _jpg(_noise_frame(1))
    detector = _Detector(tracker=_Tracker(confidence=0.9))
    _run_ladder(detector, _settled_differ(frame), frame)
    assert detector.detections == 0


def test_an_unchanged_frame_with_a_lost_tracker_keeps_the_last_entities(_cache) -> None:
    """A tracker that lost its tracks has nothing to extrapolate from."""
    _cache.set_detected_entities([{"class": "mill", "center": (5, 5)}])
    frame = _jpg(_noise_frame(1))
    detector = _Detector(tracker=_Tracker(confidence=0.1))
    held = _run_ladder(detector, _settled_differ(frame), frame)
    assert [entity.class_name for entity in held] == ["mill"]


def test_cached_detector_object_remains_renderable_on_unchanged_frame(_cache) -> None:
    """Regression: the overlay used to receive the executor cache's dictionaries."""
    from core import DetectedEntity
    from gameplay_agent.overlay import DetectionOverlay

    class _Canvas:
        def __init__(self) -> None:
            self.rectangles: list[tuple[float, float, float, float]] = []

        def delete(self, _tag: str) -> None:
            return None

        def create_rectangle(
            self, x0: float, y0: float, x1: float, y1: float, **_kwargs: object
        ) -> None:
            self.rectangles.append((x0, y0, x1, y1))

        def create_text(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Root:
        def geometry(self, _value: str) -> None:
            return None

        def update_idletasks(self) -> None:
            return None

        def update(self) -> None:
            return None

    _cache.set_detected_entities(
        [
            DetectedEntity(
                id="mill_0",
                class_name="mill",
                bbox=(10.0, 20.0, 30.0, 40.0),
                center=(20.0, 30.0),
                confidence=0.9,
            )
        ]
    )
    frame = _jpg(_noise_frame(1))
    held = _run_ladder(_Detector(tracker=_Tracker(confidence=0.1)), _settled_differ(frame), frame)
    overlay = DetectionOverlay.__new__(DetectionOverlay)
    canvas = _Canvas()
    overlay._root = _Root()
    overlay._canvas = canvas
    overlay._visible = True
    overlay._ocr_fields = {}

    overlay.show(held, (0, 0, 640, 360))

    assert canvas.rectangles[0] == (10.0, 20.0, 30.0, 40.0)


def test_a_panned_frame_is_served_from_the_cache(_cache) -> None:
    """The view moved but the map did not, so shift what we already know."""
    first = _noise_frame(3)
    _cache.set_detected_entities([{"class": "tree", "center": (100, 100)}])
    detector = _Detector(tracker=None)
    moved = _run_ladder(detector, _settled_differ(_jpg(first)), _jpg(_panned(first, 80, 0)))
    # Phase correlation is sub-pixel, not exact — the same tolerance the pan
    # measurement itself is tested to.
    assert moved[0].center == pytest.approx((180, 100), abs=_MAX_PAN_ERROR_PX)


def test_a_panned_frame_costs_no_detection(_cache) -> None:
    first = _noise_frame(3)
    _cache.set_detected_entities([{"class": "tree", "center": (100, 100)}])
    detector = _Detector(tracker=None)
    _run_ladder(detector, _settled_differ(_jpg(first)), _jpg(_panned(first, 80, 0)))
    assert detector.detections == 0


def test_an_unrelated_frame_pays_for_a_detection(_cache) -> None:
    """Two frames that cannot correlate: no rung above the detector applies."""
    detector = _Detector(entities=[{"class": "sheep"}], tracker=None)
    differ = _settled_differ(_jpg(_noise_frame(1)))
    detected = _run_ladder(detector, differ, _jpg(_noise_frame(2)))
    assert [entity.class_name for entity in detected] == ["sheep"]


def test_a_disabled_cache_always_detects(_cache, monkeypatch: pytest.MonkeyPatch) -> None:
    """AOE2_RESCAN_CACHE=false is the A/B condition — it must really disable it."""
    from gameplay_agent import detection_phase as dp

    monkeypatch.setattr(dp.config, "rescan_cache", False, raising=False)
    first = _noise_frame(3)
    _cache.set_detected_entities([{"class": "tree", "center": (100, 100)}])
    detector = _Detector(tracker=None)
    _run_ladder(detector, _settled_differ(_jpg(first)), _jpg(_panned(first, 80, 0)))
    assert detector.detections == 1


def test_a_collapsed_entity_count_resets_the_tracker(_cache) -> None:
    """Half the world vanishing means the camera moved, not that it emptied."""
    _cache.set_detected_entities([{"class": "tree"} for _ in range(10)])
    tracker = _Tracker(confidence=0.0)
    detector = _Detector(entities=[{"class": "tree"}], tracker=tracker)
    _run_ladder(detector, _settled_differ(_jpg(_noise_frame(1))), _jpg(_noise_frame(2)))
    assert tracker.resets == 1


def test_no_differ_still_detects(_cache) -> None:
    """The differ is optional — opencv may be absent on a host."""
    detector = _Detector(entities=[{"class": "sheep"}], tracker=None)
    detected = _run_ladder(detector, None, _jpg(_noise_frame(1)))
    assert [entity.class_name for entity in detected] == ["sheep"]


def test_a_failed_detection_empties_the_cache(_cache) -> None:
    """A stale cache after a failure would strand every target_class click."""
    from gameplay_agent import detection_phase as dp

    class _Broken(_Detector):
        def detect_fast(self, _screenshot: bytes) -> list[object]:
            raise RuntimeError("no model on this host")

    _cache.set_detected_entities([{"class": "tree"}])
    _run_ladder(_Broken(tracker=None), None, _jpg(_noise_frame(1)))
    assert dp.get_detected_entities() == []


def test_the_detector_is_built_without_sahi(monkeypatch) -> None:
    """`get_detector` defaults to use_sahi=True, and SAHI tiles the 3024 px frame
    into crops the model never trained on: real F1 0.04 against 0.42."""
    from gameplay_agent import detection_phase as dp

    seen: dict[str, object] = {}

    def _spy(**kwargs: object) -> object:
        seen.update(kwargs)
        return _Detector(tracker=None)

    monkeypatch.setattr(dp, "DETECTION_AVAILABLE", True)
    monkeypatch.setattr(dp.config, "detection_host", "")
    monkeypatch.setattr(dp, "get_detector", _spy)
    dp.init_detector()
    assert seen["use_sahi"] is False
