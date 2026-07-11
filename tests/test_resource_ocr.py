"""Tests for the local resource-bar reader.

Synthesizes a screenshot + calibration + glyph templates (no real game assets
needed) and drives the PUBLIC path: `read_resource_bar` →
`evaluate_resource_readings`, plus runtime auto-calibration (`autodetect_calibration`)
parity against the real `vision_fixtures/` frames, box geometry, and the
strategist's calibration precedence.

Skipped automatically where OpenCV / RapidOCR isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")  # template backend needs OpenCV

import asyncio
import io
from pathlib import Path

import numpy as np
from gameplay_agent.resource_ocr import (
    RESOURCE_FIELDS,
    Box,
    Calibration,
    FieldBox,
    _build_fields_single_frame,
    _map_age,
    _render_digit_image,
    autodetect_calibration,
    read_resource_bar,
)
from gameplay_agent.strategist_eval import (
    all_vision_fixtures,
    evaluate_resource_readings,
    load_vision_fixture,
    resolve_screenshot_path,
)
from PIL import Image

# On-screen layout we synthesize: (field, value, left-x). y is shared.
_LAYOUT = [
    ("wood", "150", 40),
    ("food", "245", 260),
    ("gold", "0", 480),
    ("stone", "200", 700),
    ("population", "8/15", 920),
]
_FIELD_Y = 12


def _png_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_bar(tmp_path):
    """Build templates + a screenshot + a matching Calibration in tmp_path."""
    tdir = tmp_path / "templates"
    tdir.mkdir()
    for d in range(10):
        Image.fromarray(_render_digit_image(str(d))).save(tdir / f"{d}.png")
    Image.fromarray(_render_digit_image("/")).save(tdir / "slash.png")

    canvas = np.zeros((60, 1200), dtype=np.uint8)  # dark bar
    fields: dict[str, FieldBox] = {}
    for name, value, x in _LAYOUT:
        glyphs = _render_digit_image(value)
        h, w = glyphs.shape
        canvas[_FIELD_Y : _FIELD_Y + h, x : x + w] = glyphs
        fields[name] = FieldBox(x - 2, _FIELD_Y - 2, x + w + 2, _FIELD_Y + h + 2)

    calib = Calibration(width=1200, height=60, fields=fields, template_dir=tdir)
    return _png_bytes(canvas), calib


def test_read_resource_bar_reads_all_fields(synthetic_bar):
    shot, calib = synthetic_bar
    readings = read_resource_bar(shot, calib, backend="template")
    assert readings["wood"] == 150
    assert readings["food"] == 245
    assert readings["gold"] == 0
    assert readings["stone"] == 200
    assert readings["population"] == "8/15"


def test_readings_pass_the_existing_scorer(synthetic_bar):
    """The reader output must score clean against the unchanged harness."""
    shot, calib = synthetic_bar
    readings = read_resource_bar(shot, calib, backend="template")
    expected = {
        "wood": {"min": 145, "max": 155},  # range = tolerance, like real fixtures
        "food": 245,  # bare int = exact
        "gold": 0,
        "stone": 200,
        "population": "8/15",
    }
    failures = evaluate_resource_readings(expected, readings)
    assert failures == [], failures


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dark Age", "Dark Age"),
        ("Feudal Age", "Feudal Age"),
        ("Castle Age", "Castle Age"),
        ("Imperial Age", "Imperial Age"),
        ("Imperial Aqe", "Imperial Age"),  # noisy suffix, keyword intact
        ("2Imperial Age", "Imperial Age"),  # leading junk
        ("CastleAge", "Castle Age"),  # missing space
        ("", ""),
        ("xyz", ""),  # no age keyword
    ],
)
def test_map_age(text, expected):
    """Age keyword mapping is robust to OCR noise (offline; no engine needed)."""
    assert _map_age(text) == expected


# ---------------------------------------------------------------------------
# Runtime auto-calibration: box geometry (no OCR engine needed)
# ---------------------------------------------------------------------------

# Detected (RapidOCR-style, tight) boxes mimicking the 3024x1964 resource bar.
_DETECTED = {
    "wood": Box(80, 177, 121, 203),
    "food": Box(234, 177, 278, 203),
    "gold": Box(388, 177, 431, 203),
    "stone": Box(544, 177, 588, 203),
}
_DETECTED_POP = Box(701, 177, 737, 202)
_DETECTED_AGE = Box(1045, 169, 1184, 209)


def test_build_fields_single_frame_geometry():
    """Boxes hug the detected number: left edge = detected x0 (keeps the icon
    out), tight bottom (excludes the sub-count row), right never reaches the next
    field and never cuts the value."""
    fields = _build_fields_single_frame(
        dict(_DETECTED), _DETECTED_POP, _DETECTED_AGE, frame_w=3024, pad=4
    )
    # Left edge anchored at the detected left edge — no left pad (the resource
    # icon sits immediately left and would OCR as a stray leading digit).
    assert fields["food"].x0 == 234
    assert fields["stone"].x0 == 544
    # Shared, tight y-band: top = min(y0) - small pad; bottom = max(y1) un-padded.
    assert fields["food"].y1 == 203  # no downward pad → excludes the sub-count
    assert fields["wood"].y0 == fields["stone"].y0  # one shared band
    assert fields["food"].y0 == 175  # 177 - _Y_TOP_PAD(2)

    boxes = {**_DETECTED, "population": _DETECTED_POP}
    order = ["wood", "food", "gold", "stone", "population"]
    for i, name in enumerate(order[:-1]):
        assert fields[name].x1 >= boxes[name][2]  # never cuts the detected value
        assert fields[name].x1 < boxes[order[i + 1]][0]  # never reaches next field
    # Rightmost field (population) just gets the right pad, capped to the frame.
    assert fields["population"].x1 == _DETECTED_POP[2] + 4
    assert "age" in fields and fields["age"].x1 >= _DETECTED_AGE[2]


def test_build_fields_caps_at_next_field_without_cutting_value():
    """When a value runs close to the next field, the right edge is clamped to the
    next field but never below the detected value (the value always fits)."""
    crowded = dict(_DETECTED)
    crowded["stone"] = Box(544, 177, 697, 203)  # wide value, only 4px before pop@701
    fields = _build_fields_single_frame(crowded, _DETECTED_POP, None, frame_w=3024, pad=4)
    assert fields["stone"].x1 >= 697  # detected value never cut
    assert fields["stone"].x1 < _DETECTED_POP[0]  # but never into population


def test_build_fields_empty_when_nothing_detected():
    assert _build_fields_single_frame({}, None, None, frame_w=3024, pad=4) == {}


def test_detect_idle_present_by_icon_colour():
    """Idle presence comes from the badge colour: yellow (saturated) = idle, grey = none.

    Anchored on the population field; validated on real frames as ~10x separated
    (grey ≤ 10 saturation, yellow ≥ 58). Here we synthesize both states.
    """
    from gameplay_agent.resource_ocr import detect_idle_present

    pop = FieldBox(700, 176, 800, 204)  # icon sampled just right of this
    frame = np.zeros((260, 1000, 3), dtype=np.uint8)

    # Grey icon (equal channels → zero saturation) → no idle villagers.
    frame[160:212, 800:880] = (120, 120, 120)
    assert detect_idle_present(frame, pop) is False

    # Bright-yellow icon (R,G high, B low → high saturation) → idle present.
    frame[160:212, 800:880] = (235, 205, 20)
    assert detect_idle_present(frame, pop) is True


def test_read_idle_count_returns_none_without_digits():
    """No white glyph strokes in the badge window → None (unknown), never a guess."""
    pytest.importorskip("cv2")
    from gameplay_agent.resource_ocr import read_idle_count

    pop = FieldBox(700, 176, 800, 204)
    frame = np.zeros((260, 1000, 3), dtype=np.uint8)
    frame[160:212, 800:880] = (235, 205, 20)  # yellow badge, no digit anywhere
    assert read_idle_count(frame, pop) is None


def test_read_idle_count_rejects_unfamiliar_shapes():
    """A white blob of digit-like size that matches no template scores below the
    NCC floor → None rather than a fabricated count."""
    pytest.importorskip("cv2")
    from gameplay_agent.resource_ocr import read_idle_count

    pop = FieldBox(700, 176, 800, 204)
    ph = pop.y1 - pop.y0  # 28px — window is pop.x0 + [3.5, 6.8]*ph
    frame = np.zeros((260, 1000, 3), dtype=np.uint8)
    # Solid white square (no digit structure) inside the count window.
    frame[182 : 182 + ph - 6, 810 : 810 + 18] = (255, 255, 255)
    assert read_idle_count(frame, pop) is None


def test_calibration_field_rects_returns_plain_tuples():
    """field_rects() exposes each FieldBox as a plain (x0,y0,x1,y1) tuple so the
    overlay can draw the reading regions without importing FieldBox."""
    calib = Calibration(
        width=1200,
        height=60,
        fields={"food": FieldBox(10, 20, 30, 40), "wood": FieldBox(50, 20, 70, 40)},
        template_dir=Path("/nonexistent"),
    )
    assert calib.field_rects() == {"food": (10, 20, 30, 40), "wood": (50, 20, 70, 40)}


# ---------------------------------------------------------------------------
# Strategist calibration precedence (no LLM, no network, no OCR engine)
# ---------------------------------------------------------------------------


def test_read_hud_readings_precedence(monkeypatch):
    """Hand YAML wins; else auto-detect; both-None → {} with no per-field read.

    This is the per-turn HUD reader the game loop calls every tick (and the
    strategist reuses); the resolution precedence is the contract under test.
    """
    from types import SimpleNamespace

    from gameplay_agent.providers import strategist as strat_mod

    png = _png_bytes(np.zeros((10, 20, 3), dtype=np.uint8))
    calls = {"auto": 0, "read": 0}

    def fake_read(_bytes, _calib, *, backend):
        calls["read"] += 1
        return {"food": 200, "wood": 200, "gold": 100, "stone": 200, "population": "4/5"}

    def fake_autodetect(_bytes):
        calls["auto"] += 1
        return SimpleNamespace(fields={"food": None}) if calls["auto"] == 1 else None

    monkeypatch.setattr(strat_mod, "read_resource_bar", fake_read)
    monkeypatch.setattr(strat_mod, "autodetect_calibration", fake_autodetect)

    # Hand YAML present → used; auto-detect never called.
    monkeypatch.setattr(
        strat_mod, "calibration_for", lambda w, h: SimpleNamespace(fields={"food": None})
    )
    out, calib = asyncio.run(strat_mod.read_hud_readings(png))
    assert out["food"] == 200 and out["population"] == "4/5"
    assert calib is not None  # the calibration is returned (for the overlay)
    assert calls["auto"] == 0 and calls["read"] == 1

    # No hand YAML → auto-detect runs (first call returns a calib) and we read.
    monkeypatch.setattr(strat_mod, "calibration_for", lambda w, h: None)
    out, _calib = asyncio.run(strat_mod.read_hud_readings(png))
    assert out["wood"] == 200 and calls["auto"] == 1

    # Auto-detect now fails to localize → {} and read_resource_bar NOT called.
    reads_before = calls["read"]
    out, calib = asyncio.run(strat_mod.read_hud_readings(png))
    assert out == {} and calib is None
    assert calls["read"] == reads_before


# ---------------------------------------------------------------------------
# Runtime auto-calibration: real-frame parity (needs the RapidOCR engine)
# ---------------------------------------------------------------------------

_REAL_FIXTURES = [p for p in all_vision_fixtures() if p.stem.startswith("real_")]


def _expected_without_lone_digits(expected: dict) -> dict:
    """Drop lone single-digit resource expectations.

    A field showing a single glyph (e.g. stone "1") produces no RapidOCR
    detection, so content-based auto-detect can't localize it — it's omitted
    (last value kept), the documented graceful-degradation case the hand YAML
    covers. Multi-digit values are unaffected.
    """
    return {k: v for k, v in expected.items() if not (k in RESOURCE_FIELDS and len(str(v)) == 1)}


@pytest.mark.parametrize("fixture_path", _REAL_FIXTURES, ids=lambda p: p.stem)
def test_autodetect_matches_hand_calibration(fixture_path):
    """The no-YAML auto path reads real frames as well as the hand calibration."""
    pytest.importorskip("rapidocr_onnxruntime")
    from gameplay_agent.providers.strategist import _clean_readings

    fixture = load_vision_fixture(fixture_path)
    data = resolve_screenshot_path(fixture_path, fixture["screenshot"]).read_bytes()
    calib = autodetect_calibration(data)
    assert calib is not None, f"{fixture_path.name}: auto-detect failed to localize the bar"
    readings = _clean_readings(read_resource_bar(data, calib, backend="rapidocr"))
    expected = _expected_without_lone_digits(fixture["expected"])
    failures = evaluate_resource_readings(expected, readings)
    assert failures == [], f"{fixture_path.name}: {failures} (got {readings})"


# Ground truth read off the badge in each fixture frame by eye (the fixture YAMLs
# predate the idle-count reader). Covers both HUD skins, the yellow and grey badge
# states, and a two-digit count.
_EXPECTED_IDLE_COUNT: dict[str, int] = {
    "real_000_dark_start": 3,
    "real_040_castle": 0,
    "real_060_imperial": 0,
    "real_090_castle": 0,
    "real_160_imperial": 18,
    "real_180_low_pop": 0,
    "real_215_imperial": 0,
}


@pytest.mark.parametrize("fixture_path", _REAL_FIXTURES, ids=lambda p: p.stem)
def test_read_idle_count_real_fixtures(fixture_path):
    """The badge count digit reads exactly on every real frame (template NCC)."""
    pytest.importorskip("rapidocr_onnxruntime")  # autodetect needs the engine
    pytest.importorskip("cv2")
    from gameplay_agent.resource_ocr import read_idle_count

    fixture = load_vision_fixture(fixture_path)
    data = resolve_screenshot_path(fixture_path, fixture["screenshot"]).read_bytes()
    calib = autodetect_calibration(data)
    assert calib is not None
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    got = read_idle_count(rgb, calib.fields["population"])
    assert got == _EXPECTED_IDLE_COUNT[fixture_path.stem], f"{fixture_path.stem}: got {got}"


def test_autodetect_no_template_path_reads_multidigit():
    """Auto-detect at a resolution with NO digit templates (the live 3024x1672
    case) still reads multi-digit values via RapidOCR alone."""
    pytest.importorskip("rapidocr_onnxruntime")
    from gameplay_agent.providers.strategist import _clean_readings

    # real_060 is all multi-digit (no lone-digit fallback needed).
    fixture_path = next(p for p in _REAL_FIXTURES if p.stem == "real_060_imperial")
    fixture = load_vision_fixture(fixture_path)
    data = resolve_screenshot_path(fixture_path, fixture["screenshot"]).read_bytes()
    calib = autodetect_calibration(data)
    assert calib is not None
    no_templates = Calibration(
        calib.width, calib.height, calib.fields, Path("/nonexistent/__no_templates__")
    )
    readings = _clean_readings(read_resource_bar(data, no_templates, backend="rapidocr"))
    assert evaluate_resource_readings(fixture["expected"], readings) == []


def test_autodetect_rejects_non_bar_frames():
    """The acceptance gate returns None (no crash) on frames with no resource bar,
    so the strategist falls back to last-known state."""
    pytest.importorskip("rapidocr_onnxruntime")
    blank = _png_bytes(np.zeros((400, 1200, 3), dtype=np.uint8))
    assert autodetect_calibration(blank) is None
    assert autodetect_calibration(b"") is None


def test_warm_up_ocr_never_raises(monkeypatch):
    """Warm-up is opportunistic: an engine failure logs and returns, no raise."""
    from gameplay_agent import resource_ocr as ro

    def _boom() -> object:
        raise RuntimeError("no engine on this host")

    monkeypatch.setattr(ro, "_rapidocr_engine", _boom)
    ro.warm_up_ocr()  # must not raise


def test_warm_up_ocr_runs_one_inference(monkeypatch):
    """Warm-up builds the engine and pushes one tiny frame through it."""
    from gameplay_agent import resource_ocr as ro

    calls: list[object] = []

    def _fake_engine() -> object:
        def engine(img: object) -> tuple[None, float]:
            calls.append(img)
            return None, 0.0

        return engine

    monkeypatch.setattr(ro, "_rapidocr_engine", _fake_engine)
    ro.warm_up_ocr()
    assert len(calls) == 1
