"""Camera-pan estimation — the basis of the static-object rescan cache.

A mid-turn rescan exists only so the next step can find a static target after
the camera jumped. Run 2026_08_22_1 paid ~2 s for each of 112 such rescans. If
the view merely panned, the cached map is still valid once translated.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from detection.inference.frame_diff import FrameChange, FrameDiffer
from gameplay_agent.detection_phase import (
    STATIC_CLASSES,
    _pan_translation,
    _translated_static,
)
from PIL import Image

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
# AOE2_RESCAN_CACHE=false is the A/B condition. A refusal logged on every rescan
# would bury the signal the comparison exists to collect.


def _drive_one_rescan(monkeypatch: pytest.MonkeyPatch, *, cache_on: bool) -> None:
    """Register the rescan callbacks against stubs, then run one rescan."""
    import asyncio
    from types import SimpleNamespace

    from detection.inference.frame_diff import FrameDiffer
    from gameplay_agent import detection_phase as dp
    from gameplay_agent import executor as ex

    monkeypatch.setattr(dp.config, "rescan_cache", cache_on, raising=False)
    monkeypatch.setattr(dp, "capture_screenshot", lambda quality=0: (_jpg(_noise_frame(2)), 0, 0))
    monkeypatch.setattr(dp, "get_game_window_rect", lambda: None)
    ex.set_detected_entities([{"class": "tree", "center": (10, 10)}])

    differ = FrameDiffer(threshold=0.03)
    differ.compare(_jpg(_noise_frame(1)))  # a previous frame, so a pan is measurable
    detector = SimpleNamespace(
        tracker=None,
        _previous_entities=[],
        detect_fast_multi=lambda _s: [],
    )
    dp._register_rescan_callbacks(detector, None, differ)  # pyright: ignore[reportArgumentType]
    rescan = ex.get_rescan_fn()
    assert rescan is not None
    asyncio.run(rescan())


def test_a_disabled_cache_logs_no_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    _drive_one_rescan(monkeypatch, cache_on=False)
    assert "rescan_pan_declined" not in capsys.readouterr().out


def test_an_enabled_cache_reports_why_it_declined(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two unrelated frames cannot correlate, so the refusal must say so."""
    capsys.readouterr()
    _drive_one_rescan(monkeypatch, cache_on=True)
    assert "low_confidence" in capsys.readouterr().out
