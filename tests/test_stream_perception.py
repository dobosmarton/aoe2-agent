"""Fresh detections, camera motion, and bounded visual identity."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from core import DetectedEntity
from gameplay_agent.camera_motion import (
    CameraMotion,
    MotionLimits,
    analysis_pixels,
    estimate_motion,
)
from gameplay_agent.detection_phase import detect_frame
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.executor import ActionLedger
from gameplay_agent.loops.snapshot import FramePipe, Perception
from gameplay_agent.loops.source import GameSource, RawSpatial, StreamCapture
from gameplay_agent.screen import NativeCapture
from gameplay_agent.temporal_entities import TemporalEntities

_SIZE = (640, 360)


def _pixels(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (360, 640, 3), dtype=np.uint8)


def _entity(label: str, box: tuple[float, float, float, float]) -> EntitySnapshot:
    x0, y0, x1, y1 = box
    return EntitySnapshot("raw", label, box, ((x0 + x1) / 2, (y0 + y1) / 2), 0.8)


def _observe(
    tracks: TemporalEntities,
    entities: tuple[EntitySnapshot, ...],
    *,
    capture_id: int,
    at: float,
    motion: CameraMotion = CameraMotion("stable"),
    pixels: np.ndarray | None = None,
) -> tuple[EntitySnapshot, ...]:
    frame = _pixels() if pixels is None else pixels
    return tracks.observe(
        entities,
        capture_id=capture_id,
        captured_at=at,
        camera_revision=0,
        motion=motion,
        rgb=frame.tobytes(),
        size=_SIZE,
    )


@pytest.mark.parametrize(("dx", "dy"), [(8, 4), (-12, 6), (0, 0)])
def test_camera_translation_is_in_capture_pixels(dx: int, dy: int) -> None:
    frame = _pixels()
    current = np.roll(frame, (dy, dx), axis=(0, 1))
    limits = MotionLimits()
    before = analysis_pixels(frame.tobytes(), *_SIZE, limits)
    after = analysis_pixels(current.tobytes(), *_SIZE, limits)

    motion = estimate_motion(before, after, full_size=_SIZE)

    assert motion.kind == ("stable" if (dx, dy) == (0, 0) else "pan")
    assert motion.shift == pytest.approx((dx, dy), abs=2)


def test_unrelated_or_untextured_views_reset_history() -> None:
    limits = MotionLimits()
    before = analysis_pixels(_pixels(1).tobytes(), *_SIZE, limits)
    unrelated = analysis_pixels(_pixels(2).tobytes(), *_SIZE, limits)
    blank = np.zeros_like(before)

    assert estimate_motion(before, unrelated, full_size=_SIZE).kind == "reset"
    assert estimate_motion(blank, before, full_size=_SIZE).kind == "reset"


def test_known_moving_region_cannot_imply_camera_motion() -> None:
    frame = _pixels()
    changed = frame.copy()
    changed[90:150, 90:150] = _pixels(2)[90:150, 90:150]
    limits = MotionLimits()
    before = analysis_pixels(frame.tobytes(), *_SIZE, limits)
    after = analysis_pixels(changed.tobytes(), *_SIZE, limits)

    motion = estimate_motion(
        before,
        after,
        full_size=_SIZE,
        excluded_boxes=((90, 90, 150, 150),),
    )

    assert motion.kind == "stable"


def test_zoom_like_scale_change_is_not_treated_as_a_translation() -> None:
    from PIL import Image

    frame = _pixels()
    image = Image.fromarray(frame)
    zoomed = image.crop((30, 20, 610, 340)).resize(_SIZE)
    limits = MotionLimits()
    before = analysis_pixels(frame.tobytes(), *_SIZE, limits)
    after = analysis_pixels(zoomed.tobytes(), *_SIZE, limits)

    assert estimate_motion(before, after, full_size=_SIZE).kind == "reset"


def test_pan_preserves_identity_but_never_predicts_a_visible_box() -> None:
    tracks = TemporalEntities()
    first = _observe(tracks, (_entity("villager", (100, 100, 120, 140)),), capture_id=1, at=1.0)
    second = _observe(
        tracks,
        (_entity("villager", (108, 104, 128, 144)),),
        capture_id=2,
        at=1.2,
        motion=CameraMotion("pan", (8, 4)),
    )

    assert second[0].id == first[0].id
    assert second[0].bbox == (108, 104, 128, 144)
    assert _observe(tracks, (), capture_id=3, at=1.4) == ()


def test_class_flicker_does_not_authorize_historical_click() -> None:
    tracks = TemporalEntities()
    box = (100, 100, 120, 140)
    _observe(tracks, (_entity("sheep", box),), capture_id=1, at=1.0)
    _observe(tracks, (_entity("sheep", box),), capture_id=2, at=1.2)
    flicker = _observe(tracks, (_entity("villager", box),), capture_id=3, at=1.3)

    assert flicker[0].class_name == "villager"
    assert flicker[0].raw_class_name == "villager"
    assert flicker[0].class_agreement is not None
    assert flicker[0].class_agreement < 1.0


def test_history_expires_and_track_ids_are_not_reused() -> None:
    tracks = TemporalEntities()
    box = (100, 100, 120, 140)
    first = _observe(tracks, (_entity("tree", box),), capture_id=1, at=1.0)
    second = _observe(tracks, (_entity("tree", box),), capture_id=2, at=2.1)

    assert first[0].id != second[0].id
    assert second[0].class_agreement is None


def test_occlusion_keeps_a_short_lived_identity_without_publishing_prediction() -> None:
    tracks = TemporalEntities()
    box = (100, 100, 120, 140)
    first = _observe(tracks, (_entity("villager", box),), capture_id=1, at=1.0)
    invisible = _observe(tracks, (), capture_id=2, at=1.2)
    third = _observe(tracks, (_entity("villager", box),), capture_id=3, at=1.4)

    assert invisible == ()
    assert third[0].id == first[0].id


def test_ambiguous_crossing_starts_a_new_track_instead_of_guessing() -> None:
    tracks = TemporalEntities()
    pixels = np.zeros((360, 640, 3), dtype=np.uint8)
    first = _observe(
        tracks,
        (_entity("villager", (100, 100, 120, 140)), _entity("villager", (130, 100, 150, 140))),
        capture_id=1,
        at=1.0,
        pixels=pixels,
    )
    crossing = _observe(
        tracks,
        (_entity("villager", (115, 100, 135, 140)),),
        capture_id=2,
        at=1.2,
        pixels=pixels,
    )

    assert crossing[0].id not in {item.id for item in first}
    assert tracks.last_ambiguous_count == 1


def test_repeated_label_evidence_can_correct_an_initial_error() -> None:
    tracks = TemporalEntities()
    box = (100, 100, 120, 140)
    _observe(tracks, (_entity("villager", box),), capture_id=1, at=1.0)
    _observe(tracks, (_entity("sheep", box),), capture_id=2, at=1.2)
    corrected = _observe(tracks, (_entity("sheep", box),), capture_id=3, at=1.4)

    assert corrected[0].class_name == "sheep"
    assert corrected[0].raw_class_name == "sheep"


def test_camera_jump_resets_history_without_reusing_an_id() -> None:
    tracks = TemporalEntities()
    box = (100, 100, 120, 140)
    first = _observe(tracks, (_entity("tree", box),), capture_id=1, at=1.0)
    second = _observe(
        tracks,
        (_entity("tree", box),),
        capture_id=2,
        at=1.2,
        motion=CameraMotion("reset", reason="jump"),
    )

    assert second[0].id != first[0].id
    assert _observe(tracks, (_entity("tree", box),), capture_id=1, at=1.0) == ()


def test_uncommanded_pan_is_discovered_when_spatial_result_is_published() -> None:
    ledger = ActionLedger()
    source = GameSource(ledger=ledger)
    first_pixels = _pixels()
    second_pixels = np.roll(first_pixels, (4, 8), axis=(0, 1))
    limits = MotionLimits()

    def raw(
        identifier: int, at: float, pixels: np.ndarray, box: tuple[float, float, float, float]
    ) -> RawSpatial:
        capture = StreamCapture(
            NativeCapture(pixels.tobytes(), 640, 360, at, (0, 0, 640, 360)),
            identifier,
            0,
            0,
            0,
            0,
            0,
        )
        analysis = analysis_pixels(pixels.tobytes(), *_SIZE, limits)
        return RawSpatial(capture, b"jpeg", (_entity("tree", box),), analysis.tobytes(), {})

    async def drive() -> tuple[EntitySnapshot, EntitySnapshot]:
        first = await source.integrate_spatial(raw(1, 1.0, first_pixels, (100, 100, 120, 140)))
        second = await source.integrate_spatial(raw(2, 1.2, second_pixels, (108, 104, 128, 144)))
        assert first is not None and second is not None
        return first.frame.entities[0], second.frame.entities[0]

    first, second = asyncio.run(drive())
    assert ledger.camera_revision == 1
    assert second.id == first.id
    assert second.bbox == (108, 104, 128, 144)


def test_same_frame_always_invokes_detector() -> None:
    class Detector:
        def __init__(self) -> None:
            self.calls = 0

        def detect_fast(self, _image: bytes) -> list[DetectedEntity]:
            self.calls += 1
            return [DetectedEntity("raw", "tree", (10, 10, 20, 30), (15, 20), 0.9)]

    detector = Detector()
    first = asyncio.run(detect_frame(detector, b"same"))  # type: ignore[arg-type]
    second = asyncio.run(detect_frame(detector, b"same"))  # type: ignore[arg-type]

    assert detector.calls == 2
    assert first == second


def test_hud_and_spatial_publications_keep_independent_versions() -> None:
    pipe = FramePipe()
    spatial = pipe.put_spatial(
        Perception(
            capture_id=1,
            spatial_capture_id=1,
            captured_at=10.0,
            spatial_captured_at=10.0,
            window_rect=(10, 20, 640, 360),
        )
    )
    hud = pipe.put_hud(
        Perception(
            capture_id=2,
            hud_capture_id=2,
            captured_at=10.1,
            hud_captured_at=10.1,
            hud_readings={"food": 200},
            window_rect=(10, 20, 640, 360),
        )
    )

    assert spatial.spatial_version == 1
    assert hud.spatial_version == 1
    assert hud.hud_version == 1
    assert hud.spatial_captured_at == 10.0
    assert hud.hud_captured_at == 10.1
    assert (hud.spatial_capture_id, hud.hud_capture_id) == (1, 2)
    assert hud.window_rect == (10, 20, 640, 360)


def test_two_consumers_are_woken_by_one_publication() -> None:
    async def drive() -> tuple[Perception, Perception]:
        pipe = FramePipe()
        first = asyncio.create_task(pipe.after(0.0))
        second = asyncio.create_task(pipe.after(0.0))
        await asyncio.sleep(0)
        pipe.put(Perception(captured_at=1.0))
        return await asyncio.gather(first, second)

    first, second = asyncio.run(drive())
    assert first is second


def test_spatial_publication_after_newer_hud_still_wakes_readers() -> None:
    async def drive() -> Perception:
        pipe = FramePipe()
        hud = pipe.put_hud(
            Perception(captured_at=10.2, hud_captured_at=10.2, hud_readings={"food": 200})
        )
        waiting = asyncio.create_task(pipe.after_version(hud.publication_version))
        await asyncio.sleep(0)
        pipe.put_spatial(
            Perception(
                captured_at=10.1,
                spatial_captured_at=10.1,
                entities=(_entity("villager", (10, 10, 20, 30)),),
                alarm=True,
            )
        )
        return await asyncio.wait_for(waiting, timeout=0.1)

    published = asyncio.run(drive())
    assert published.alarm
    assert published.captured_at == 10.2
    assert published.spatial_captured_at == 10.1
    assert published.publication_version == 2
