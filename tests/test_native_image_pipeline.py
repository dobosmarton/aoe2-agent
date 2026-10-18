"""Native pixels feed live readers while encoded images remain portable artifacts."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from detection.inference.ownership import Owner, classify_entities, classify_entities_pixels
from gameplay_agent.loops.source import GameSource, StreamCapture
from gameplay_agent.providers.strategist import read_hud_pixels
from gameplay_agent.resource_ocr import (
    calibration_for,
    decode_rgb_pixels,
    prepare_hud_pixels,
    read_resource_bar,
    read_resource_bar_pixels,
    read_selected_unit,
    read_selected_unit_pixels,
)
from gameplay_agent.screen import NativeCapture
from gameplay_agent.turn_timing import TickTimings
from PIL import Image

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def test_native_capture_exposes_read_only_rgb_and_hud_views() -> None:
    capture = NativeCapture(bytes((10, 20, 30, 40, 50, 60)), 2, 1, 1.0, (7, 9, 2, 1))
    assert capture.pixels().tolist() == [[[10, 20, 30], [40, 50, 60]]]
    assert capture.hud_pixels().shape == (1, 2, 3)
    assert not capture.pixels().flags.writeable
    assert not capture.hud_pixels().flags.writeable


def test_hud_and_selection_pixel_readers_match_lossless_decoded_input() -> None:
    fixture = Path(__file__).parents[1] / "apps/agent/src/vision_fixtures/real_180_low_pop.jpg"
    encoded = fixture.read_bytes()
    rgb = decode_rgb_pixels(encoded)
    size = (int(rgb.shape[1]), int(rgb.shape[0]))
    calibration = calibration_for(*size)
    assert calibration is not None
    hud = prepare_hud_pixels(rgb, size)
    assert (hud.width, hud.height) == size
    assert hud.gray.shape[0] == 300
    assert read_resource_bar_pixels(hud, calibration, backend="template") == read_resource_bar(
        encoded, calibration, backend="template"
    )
    assert read_selected_unit_pixels(rgb, calibration) == read_selected_unit(encoded, calibration)


def test_native_ownership_matches_the_encoded_adapter_for_identical_pixels() -> None:
    pixels = np.zeros((80, 80, 3), dtype=np.uint8)
    pixels[20:55, 20:55] = (230, 20, 20)

    class Unit:
        id = "red-unit"
        class_name = "militia_line"
        bbox = (20.0, 20.0, 55.0, 55.0)

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    expected = classify_entities(
        buffer.getvalue(), [Unit()], frozenset({"militia_line"}), {"red": Owner.ALLY}
    )
    assert (
        classify_entities_pixels(pixels, [Unit()], frozenset({"militia_line"}), {"red": Owner.ALLY})
        == expected
    )


def test_live_hud_reader_does_not_decode_an_encoded_image(monkeypatch: MonkeyPatch) -> None:
    fixture = (
        Path(__file__).parents[1] / "apps/agent/src/vision_fixtures/real_1672_dark_sep27_hud.png"
    )
    rgb = decode_rgb_pixels(fixture.read_bytes())

    def unexpected_decode(_image: bytes) -> np.ndarray:
        raise AssertionError("live HUD reads must use capture pixels")

    monkeypatch.setattr("gameplay_agent.resource_ocr.decode_rgb_pixels", unexpected_decode)
    monkeypatch.setattr("gameplay_agent.providers.strategist.decode_rgb_pixels", unexpected_decode)
    readings, calibration = asyncio.run(
        read_hud_pixels(rgb, full_size=(3024, 1672), mode="critical")
    )
    assert calibration is not None
    assert readings["villagers"] == 4


def test_full_refresh_encodes_one_jpeg_and_no_internal_png(monkeypatch: MonkeyPatch) -> None:
    pixels = NativeCapture(bytes(800 * 600 * 3), 800, 600, 1.0, (0, 0, 800, 600))
    source = GameSource()

    async def capture() -> StreamCapture:
        return StreamCapture(pixels, 1, 0, 0, 0, 0, 0)

    async def hud(*_args: object) -> dict[str, int]:
        return {"food": 200}

    calls: list[str | None] = []
    original_save = Image.Image.save

    def counted_save(
        image: Image.Image, fp: object, format: str | None = None, **params: object
    ) -> None:
        calls.append(format)
        original_save(image, fp, format=format, **params)

    monkeypatch.setattr(source, "capture_stream", capture)
    monkeypatch.setattr(source, "_hud", hud)
    monkeypatch.setattr(Image.Image, "save", counted_save)
    refresh = asyncio.run(source.capture_spatial(TickTimings()))

    assert refresh.spatial_valid
    assert calls == ["JPEG"]


def test_busy_full_refresh_reuses_its_request_jpeg(monkeypatch: MonkeyPatch) -> None:
    from gameplay_agent.detection_phase import DetectionBusyError

    pixels = NativeCapture(bytes(800 * 600 * 3), 800, 600, 1.0, (0, 0, 800, 600))
    source = GameSource()
    monkeypatch.setattr(source, "_detector", object())

    async def capture() -> StreamCapture:
        return StreamCapture(pixels, 1, 0, 0, 0, 0, 0)

    async def hud(*_args: object) -> dict[str, int]:
        return {"food": 200}

    async def busy(*_args: object) -> None:
        raise DetectionBusyError("at capacity")

    encodings = 0
    original_jpeg = NativeCapture.jpeg

    def counted_jpeg(capture: NativeCapture) -> bytes:
        nonlocal encodings
        encodings += 1
        return original_jpeg(capture)

    monkeypatch.setattr(source, "capture_stream", capture)
    monkeypatch.setattr(source, "_hud", hud)
    monkeypatch.setattr("gameplay_agent.loops.source.detect_frame", busy)
    monkeypatch.setattr(NativeCapture, "jpeg", counted_jpeg)
    refresh = asyncio.run(source.capture_spatial(TickTimings()))

    assert not refresh.spatial_valid
    assert refresh.screenshot is not None
    assert encodings == 1


def test_background_capture_does_not_encode_diagnostics(monkeypatch: MonkeyPatch) -> None:
    pixels = NativeCapture(bytes(64 * 36 * 3), 64, 36, 1.0, (0, 0, 64, 36))
    source = GameSource()

    def forbidden_jpeg(_capture: NativeCapture) -> bytes:
        raise AssertionError("capture lane must not encode")

    monkeypatch.setattr("gameplay_agent.loops.source.capture_native", lambda: pixels)
    monkeypatch.setattr(NativeCapture, "jpeg", forbidden_jpeg)
    captured = asyncio.run(source.capture_stream())
    assert captured.pixels is pixels


def test_sample_uses_the_tenth_published_spatial_jpeg(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from gameplay_agent.config import config
    from gameplay_agent.loops.source import RawSpatial

    class Writer:
        def __init__(self) -> None:
            self.saved: list[tuple[Path, bytes]] = []

        def submit(self, path: Path, data: bytes) -> bool:
            self.saved.append((path, data))
            return True

    writer = Writer()
    source = GameSource(screenshots_dir=tmp_path)
    monkeypatch.setattr(source, "_writer", writer)
    monkeypatch.setattr(config, "save_screenshots", True)
    source._published_spatial = 8

    def raw(capture_id: int, captured_at: float) -> RawSpatial:
        pixels = NativeCapture(bytes(64 * 36 * 3), 64, 36, captured_at, (0, 0, 64, 36))
        capture = StreamCapture(pixels, capture_id, 0, 0, 0, 0, 0)
        return RawSpatial(capture, b"already-encoded", (), bytes(320 * 180 * 4), {})

    async def publish_twice() -> None:
        assert await source.integrate_spatial(raw(11, 1.0)) is not None
        assert source._published_spatial == 9
        assert writer.saved == []
        assert await source.integrate_spatial(raw(12, 1.2)) is not None

    asyncio.run(publish_twice())
    assert source._published_spatial == 10
    assert len(writer.saved) == 1
    assert writer.saved[0][1] == b"already-encoded"
