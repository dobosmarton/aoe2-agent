"""Where a frame comes from, and where an action goes.

Two seams, so the clocks are not welded to one environment: the real game behind
them here, `world_sim` behind them in plan 5.3, lists behind them in a test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import structlog

from ..camera_motion import (
    MOBILE_CLASSES,
    MotionLimits,
    analysis_pixels,
    cursor_exclusions,
    estimate_motion,
)
from ..config import config
from ..detection_phase import DetectionBusyError, detect_frame, summarize_frame
from ..entity_snapshot import snapshot_entities
from ..entity_utils import build_entity_summary
from ..executor import execute_actions
from ..providers.strategist import read_hud_readings
from ..resource_ocr import calibration_for, read_selected_unit
from ..screen import (
    NativeCapture,
    capture_native,
    capture_screenshot_with_native_hud,
    save_screenshot,
)
from ..temporal_entities import TemporalEntities
from ..turn_timing import TickTimings
from ..window import get_game_window_rect
from .snapshot import FramePipe, Perception, SpatialRefresh

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from detection.inference.detector import EntityDetector
    from detection.inference.ownership import Owner
    from detection.inference.remote_detector import RemoteDetector

    from ..diagnostic_writer import DiagnosticWriter
    from ..entity_snapshot import EntitySnapshot
    from ..executor import ActionLedger, ActionResult
    from ..game_profile import GameProfile
    from ..models import Action
    from ..overlay import DetectionOverlay
    from ..resource_ocr import ResourceReadings

    # Mirrors `detection_phase.Detector` — local weights or the remote server.
    Detector: TypeAlias = EntityDetector | RemoteDetector

log = structlog.stdlib.get_logger()

# Frames between screenshot saves. Saving every frame was affordable at ~10 s
# per turn; the perceive loop runs far more often, and the write blocks it.
_SCREENSHOT_SAMPLE = 10
# A dependent click cannot wait indefinitely for fresh spatial evidence.
_REFRESH_TIMEOUT = 3.0


def frame_refresh(
    frames: FramePipe, *, selection_only: bool = False
) -> Callable[[], Awaitable[bool]]:
    """Wait for a post-input view, optionally skipping expensive detection."""

    async def refresh() -> bool:
        asked = time.monotonic()
        request = frames.request_spatial_refresh(selection_only=selection_only)
        try:
            result = await asyncio.wait_for(request, timeout=_REFRESH_TIMEOUT)
        except TimeoutError:
            log.warning("frame_refresh_timed_out", seconds=_REFRESH_TIMEOUT)
            return False
        finally:
            frames.clear_spatial_refresh(request)
        if result.captured_at <= asked or not result.spatial_valid:
            log.warning(
                "frame_refresh_rejected",
                reason="old_capture" if result.captured_at <= asked else "input_changed",
                input_revision=result.input_revision,
            )
            return False
        log.info(
            "frame_refresh_completed",
            waited_ms=round((time.monotonic() - asked) * 1000),
            input_revision=result.input_revision,
        )
        return True

    return refresh


@dataclass(frozen=True, slots=True)
class Sighting:
    """One perception pass. Ownership rides alongside the frame because only the
    alarm check reads it, and re-classifying cost 15 s (run 2026-08-20)."""

    frame: Perception
    ownership: Mapping[str, tuple[Owner, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ownership", MappingProxyType(dict(self.ownership)))


@dataclass(frozen=True, slots=True)
class StreamCapture:
    pixels: NativeCapture
    capture_id: int
    input_revision: int
    camera_revision: int
    camera_command_revision: int
    economic_revision: int
    selection_revision: int


@dataclass(frozen=True, slots=True)
class RawSpatial:
    """Worker result only; publishing owns camera and track state changes."""

    capture: StreamCapture
    screenshot: bytes
    entities: tuple[EntitySnapshot, ...]
    analysis: bytes
    ownership: Mapping[str, tuple[Owner, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ownership", MappingProxyType(dict(self.ownership)))


class FrameSource(Protocol):
    """Perception, whatever is behind it."""

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        """One frame. Records its own `capture`/`ocr`/`detect` phases."""
        ...

    async def capture_hud(self, tick: int, timings: TickTimings) -> Sighting | None:
        """A quick HUD-only observation, when supported by this source."""
        ...

    async def capture_spatial(
        self, timings: TickTimings, *, selection_only: bool = False
    ) -> SpatialRefresh:
        """Capture after input; skip detection when only selection/HUD is needed."""
        ...

    def close(self) -> None:
        """Release whatever the source owns. Never raises."""
        ...


class Actuator(Protocol):
    """Action, whatever is behind it."""

    async def execute(self, actions: Sequence[Action | dict[str, object]]) -> list[ActionResult]:
        """Run the actions in order and report what each one did."""
        ...


def _grab() -> tuple[bytes, bytes, int, int, float]:
    """Screenshot plus its capture instant, off the event loop. The stamp comes
    first, so a frame reads older than it is — staleness then errs to skipping."""
    stamped = time.monotonic()
    screenshot, native_hud, width, height = capture_screenshot_with_native_hud()
    return screenshot, native_hud, width, height, stamped


class GameSource:
    """The real game: mss, YOLO and local OCR."""

    def __init__(
        self,
        detector: Detector | None = None,
        overlay: DetectionOverlay | None = None,
        screenshots_dir: Path | None = None,
        ledger: ActionLedger | None = None,
        profile: GameProfile | None = None,
        writer: DiagnosticWriter | None = None,
    ) -> None:
        self._detector = detector
        self._overlay = overlay
        self._screenshots_dir = screenshots_dir
        self._ledger = ledger
        self._profile = profile
        self._writer = writer
        self._inference_lock = asyncio.Lock()
        self._overlay_lock = asyncio.Lock()
        self._history = TemporalEntities()
        self._last_analysis: np.ndarray | None = None
        self._last_camera_command_revision = -1
        self._last_geometry: tuple[int, int, tuple[int, int, int, int] | None] | None = None
        self._last_spatial_at = -1.0
        self._last_moving_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self._capture_id = 0
        self._published_spatial = 0
        self._discarded_busy = 0
        self._discarded_camera = 0
        self._discarded_late = 0
        self._camera_resets = 0
        self._camera_pans = 0

    async def capture_stream(self) -> StreamCapture:
        started_at = time.monotonic()
        async with self._overlay_lock:
            if self._overlay:
                self._overlay.hide()
            pixels = await asyncio.to_thread(capture_native)
        log.debug(
            "native_capture_completed",
            capture_ms=round((time.monotonic() - started_at) * 1000, 1),
            captured_at=pixels.captured_at,
        )
        self._capture_id += 1
        if (
            self._writer is not None
            and config.save_screenshots
            and self._screenshots_dir is not None
            and self._capture_id % _SCREENSHOT_SAMPLE == 0
        ):
            screenshot = await asyncio.to_thread(pixels.jpeg)
            path = self._screenshots_dir / f"capture_{self._capture_id:06d}.jpg"
            self._writer.submit(path, screenshot)
        ledger = self._ledger
        return StreamCapture(
            pixels,
            self._capture_id,
            ledger.input_revision if ledger else 0,
            ledger.camera_revision if ledger else 0,
            ledger.camera_command_revision if ledger else 0,
            ledger.economic_revision if ledger else 0,
            ledger.selection_revision if ledger else 0,
        )

    async def read_stream_hud(self, capture: StreamCapture, tick: int) -> Sighting | None:
        timings = TickTimings()
        hud = await asyncio.to_thread(capture.pixels.native_hud)
        readings = await self._hud(
            hud, tick, timings, (capture.pixels.width, capture.pixels.height)
        )
        ledger = self._ledger
        if ledger is not None and capture.economic_revision != ledger.economic_revision:
            log.info("hud_result_discarded", capture_id=capture.capture_id, reason="economic_input")
            return None
        return Sighting(
            Perception(
                hud_readings=readings,
                input_revision=capture.input_revision,
                camera_revision=capture.camera_revision,
                economic_revision=capture.economic_revision,
                captured_at=capture.pixels.captured_at,
                hud_captured_at=capture.pixels.captured_at,
                tick=tick,
                hud_only=True,
                spatial_valid=False,
                capture_id=capture.capture_id,
                hud_capture_id=capture.capture_id,
                window_rect=capture.pixels.window_rect,
            )
        )

    async def detect_stream(self, capture: StreamCapture, tick: int) -> RawSpatial | None:
        """Run one detector call without touching published camera or track state."""
        del tick
        async with self._inference_lock:
            started_at = time.monotonic()
            screenshot = await asyncio.to_thread(capture.pixels.jpeg)
            encoded_at = time.monotonic()
            try:
                entities = await detect_frame(self._detector, screenshot) if self._detector else ()
            except DetectionBusyError:
                self._discarded_busy += 1
                log.info("spatial_result_discarded", capture_id=capture.capture_id, reason="busy")
                return None
            detected_at = time.monotonic()
            pixels = capture.pixels
            analysis = await asyncio.to_thread(
                analysis_pixels, pixels.rgb, pixels.width, pixels.height, MotionLimits()
            )
            analyzed_at = time.monotonic()
            _summary, ownership = await summarize_frame(entities, screenshot, profile=self._profile)
            log.info(
                "spatial_worker_latency",
                capture_id=capture.capture_id,
                encode_ms=round((encoded_at - started_at) * 1000, 1),
                detector_ms=round((detected_at - encoded_at) * 1000, 1),
                analysis_ms=round((analyzed_at - detected_at) * 1000, 1),
                ownership_ms=round((time.monotonic() - analyzed_at) * 1000, 1),
            )
            return RawSpatial(capture, screenshot, entities, analysis.tobytes(), ownership)

    async def integrate_spatial(self, raw: RawSpatial) -> Sighting | None:
        """Publish a worker result only if its capture still matches this view."""
        capture = raw.capture
        pixels = capture.pixels
        ledger = self._ledger
        if ledger is not None and ledger.camera_command_revision != capture.camera_command_revision:
            self._discarded_camera += 1
            log.info(
                "spatial_result_discarded", capture_id=capture.capture_id, reason="camera_input"
            )
            return None
        if not self._history.accepts(capture.capture_id, pixels.captured_at):
            self._discarded_late += 1
            log.info(
                "spatial_result_discarded", capture_id=capture.capture_id, reason="older_capture"
            )
            return None
        async with self._overlay_lock:
            # Capture and overlay cannot overlap; recheck after acquiring the lock.
            if (
                ledger is not None
                and ledger.camera_command_revision != capture.camera_command_revision
            ):
                self._discarded_camera += 1
                return None
            geometry = (pixels.width, pixels.height, pixels.window_rect)
            reset = (
                self._last_geometry != geometry
                or self._last_camera_command_revision != capture.camera_command_revision
                or (self._last_spatial_at >= 0 and pixels.captured_at - self._last_spatial_at > 1.0)
            )
            previous = None if reset else self._last_analysis
            limits = MotionLimits()
            analysis = np.frombuffer(raw.analysis, dtype=np.float32).reshape(
                (limits.analysis_height, limits.analysis_width)
            )
            current_moving_boxes = tuple(
                item.bbox
                for item in raw.entities
                if item.class_name in MOBILE_CLASSES and item.bbox is not None
            )
            cursor_box = cursor_exclusions(pixels.cursor, limits)
            camera_started = time.monotonic()
            motion = estimate_motion(
                previous,
                analysis,
                full_size=(pixels.width, pixels.height),
                limits=limits,
                excluded_boxes=(*self._last_moving_boxes, *current_moving_boxes, *cursor_box),
            )
            camera_ms = (time.monotonic() - camera_started) * 1000
            self._camera_resets += motion.kind == "reset"
            self._camera_pans += motion.kind == "pan"
            if motion.kind != "stable" and self._last_spatial_at >= 0 and ledger is not None:
                ledger.camera_revision += 1
                ledger.spatial_valid = False
            revision = ledger.camera_revision if ledger else capture.camera_revision
            association_started = time.monotonic()
            aligned = self._history.observe(
                raw.entities,
                capture_id=capture.capture_id,
                captured_at=pixels.captured_at,
                camera_revision=revision,
                motion=motion,
                rgb=pixels.rgb,
                size=(pixels.width, pixels.height),
            )
            association_ms = (time.monotonic() - association_started) * 1000
            self._last_analysis = analysis
            self._last_camera_command_revision = capture.camera_command_revision
            self._last_geometry = geometry
            self._last_spatial_at = pixels.captured_at
            self._last_moving_boxes = current_moving_boxes
            fresh_raw = (item for item in raw.entities if item.bbox is not None)
            ownership = {
                current.id: raw.ownership[detected.id]
                for current, detected in zip(aligned, fresh_raw, strict=True)
                if detected.id in raw.ownership
            }
            summary = build_entity_summary(list(aligned), ownership_results=ownership)
            if self._overlay:
                self._overlay.show(aligned, get_game_window_rect())
            self._published_spatial += 1
            log.info(
                "fresh_detection_published",
                capture_id=capture.capture_id,
                count=len(aligned),
                motion=motion.kind,
                age_ms=round((time.monotonic() - pixels.captured_at) * 1000),
                camera_ms=round(camera_ms, 1),
                association_ms=round(association_ms, 1),
            )
            return Sighting(
                Perception(
                    screenshot=raw.screenshot,
                    width=pixels.width,
                    height=pixels.height,
                    entities=aligned,
                    entity_summary=summary,
                    input_revision=capture.input_revision,
                    camera_revision=revision,
                    economic_revision=capture.economic_revision,
                    selection_revision=capture.selection_revision,
                    spatial_captured_at=pixels.captured_at,
                    captured_at=pixels.captured_at,
                    tick=capture.capture_id,
                    capture_id=capture.capture_id,
                    spatial_capture_id=capture.capture_id,
                    window_rect=pixels.window_rect,
                ),
                ownership,
            )

    async def capture_hud(self, tick: int, timings: TickTimings) -> Sighting | None:
        """Publish economic facts before the more expensive full detector pass."""
        ledger = self._ledger
        revision = ledger.input_revision if ledger is not None else 0
        camera_revision = ledger.camera_revision if ledger is not None else 0
        economic_revision = ledger.economic_revision if ledger is not None else 0
        screenshot, native_hud, width, height, captured_at = await self._screen(None, timings)
        hud_readings = await self._hud(native_hud, tick, timings, (width, height))
        valid = ledger is None or ledger.economic_revision == economic_revision
        if not valid:
            return None
        return Sighting(
            Perception(
                screenshot=screenshot,
                width=width,
                height=height,
                hud_readings=hud_readings,
                tick=tick,
                captured_at=captured_at,
                input_revision=revision,
                camera_revision=camera_revision,
                economic_revision=economic_revision,
                spatial_valid=False,
                hud_only=True,
            )
        )

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        ledger = self._ledger
        revision = ledger.input_revision if ledger is not None else 0
        camera_revision = ledger.camera_revision if ledger is not None else 0
        economic_revision = ledger.economic_revision if ledger is not None else 0
        screenshot, native_hud, width, height, captured_at = await self._screen(tick, timings)
        hud_readings, entity_result = await asyncio.gather(
            self._hud(native_hud, tick, timings, (width, height)),
            self._entities(screenshot, timings),
        )
        entities, entity_summary, ownership = entity_result
        spatial_valid = ledger is None or ledger.camera_revision == camera_revision
        return Sighting(
            frame=Perception(
                screenshot=screenshot,
                width=width,
                height=height,
                entities=tuple(entities),
                entity_summary=entity_summary,
                hud_readings=hud_readings,
                tick=tick,
                captured_at=captured_at,
                input_revision=revision,
                camera_revision=camera_revision,
                economic_revision=economic_revision,
                spatial_valid=spatial_valid,
            ),
            ownership=ownership,
        )

    async def capture_spatial(
        self, timings: TickTimings, *, selection_only: bool = False
    ) -> SpatialRefresh:
        """One native post-input capture; a requested detector refresh has priority."""
        capture = await self.capture_stream()
        pixels = capture.pixels
        screenshot = await asyncio.to_thread(pixels.jpeg)
        native_hud = await asyncio.to_thread(pixels.native_hud)
        calibration = calibration_for(pixels.width, pixels.height)

        async def selected() -> str | None:
            if calibration is None:
                return None
            return await asyncio.to_thread(read_selected_unit, screenshot, calibration)

        if selection_only:
            hud_readings, selected_unit = await asyncio.gather(
                self._hud(native_hud, 0, timings, (pixels.width, pixels.height)), selected()
            )
            entities: tuple[EntitySnapshot, ...] | None = None
            sighting: Sighting | None = None
            observed_camera_revision = capture.camera_revision
        else:
            hud_readings, selected_unit, raw = await asyncio.gather(
                self._hud(native_hud, 0, timings, (pixels.width, pixels.height)),
                selected(),
                self.detect_stream(capture, 0),
            )
            sighting = await self.integrate_spatial(raw) if raw is not None else None
            entities = snapshot_entities(sighting.frame.entities) if sighting is not None else None
            if sighting is not None:
                sighting = Sighting(
                    replace(
                        sighting.frame,
                        hud_readings=hud_readings,
                        hud_captured_at=pixels.captured_at,
                        hud_capture_id=capture.capture_id,
                    ),
                    sighting.ownership,
                )
            observed_camera_revision = (
                sighting.frame.camera_revision if sighting is not None else capture.camera_revision
            )
        spatial_valid = self._ledger is None or (
            self._ledger.camera_revision == observed_camera_revision
        )
        if entities is None and not selection_only:
            spatial_valid = False
        log.info(
            "spatial_refresh_captured",
            entity_count=len(entities) if entities is not None else 0,
            input_revision=capture.input_revision,
            spatial_valid=spatial_valid,
            selected_unit=selected_unit,
            selection_only=selection_only,
        )
        return SpatialRefresh(
            captured_at=pixels.captured_at,
            input_revision=capture.input_revision,
            spatial_valid=spatial_valid,
            hud_readings=hud_readings,
            selected_unit=selected_unit,
            screenshot=screenshot,
            entities=entities,
            sighting=sighting,
        )

    async def _screen(
        self, tick: int | None, timings: TickTimings
    ) -> tuple[bytes, bytes, int, int, float]:
        """Grab the frame. The overlay hides first, so it stays out of the shot."""
        with timings.phase("capture"):
            if self._overlay:
                self._overlay.hide()
            screenshot, native_hud, width, height, captured_at = await asyncio.to_thread(_grab)
            if tick is not None:
                self._save_sample(screenshot, tick)
        return screenshot, native_hud, width, height, captured_at

    async def _hud(
        self,
        native_hud: bytes,
        tick: int,
        timings: TickTimings,
        full_size: tuple[int, int],
    ) -> ResourceReadings:
        """Read the resource bar, and show the OCR boxes it calibrated."""
        with timings.phase("ocr"):
            hud_readings, calib = await read_hud_readings(
                native_hud,
                turn=tick,
                full_size=full_size,
                mode="critical" if tick == 0 else "routine",
            )
        if self._overlay is not None and calib is not None:
            self._overlay.set_ocr_fields(calib.field_rects())
        return hud_readings

    async def _entities(
        self, screenshot: bytes, timings: TickTimings
    ) -> tuple[list[EntitySnapshot], str, Mapping[str, tuple[Owner, float]]]:
        """Detect, then tag ownership. Empty without a detector."""
        with timings.phase("detect"):
            entities = await self._detect_entities(screenshot)
            entity_summary, ownership = await summarize_frame(
                entities, screenshot, profile=self._profile
            )
        return entities, entity_summary, ownership

    async def _detect_entities(self, screenshot: bytes) -> list[EntitySnapshot]:
        entities: list[EntitySnapshot] = []
        if self._detector:
            entities = list(
                await detect_frame(
                    self._detector,
                    screenshot,
                )
            )
        if self._overlay is not None:
            self._overlay.show(entities, get_game_window_rect())
        return entities

    def _save_sample(self, screenshot: bytes, tick: int) -> None:
        """Keep one frame in `_SCREENSHOT_SAMPLE`, for the run's image trail."""
        if not (config.save_screenshots and self._screenshots_dir):
            return
        if tick % _SCREENSHOT_SAMPLE:
            return
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = self._screenshots_dir / f"{stamp}_{tick:05d}.jpg"
        if self._writer is not None:
            self._writer.submit(path, screenshot)
        else:
            save_screenshot(screenshot, str(path))

    def close(self) -> None:
        log.info(
            "perception_stream_summary",
            captured=self._capture_id,
            published_spatial=self._published_spatial,
            discarded_busy=self._discarded_busy,
            discarded_camera=self._discarded_camera,
            discarded_late=self._discarded_late,
            camera_resets=self._camera_resets,
            camera_pans=self._camera_pans,
        )
        if self._overlay:
            self._overlay.close()

    async def aclose(self) -> None:
        """Release the overlay and remote HTTP connection after workers drain."""
        self.close()
        if self._detector is None:
            return
        from detection.inference.remote_detector import RemoteDetector

        if isinstance(self._detector, RemoteDetector):
            await self._detector.close()


class GameActuator:
    """The real game: synthetic mouse and keyboard through pyautogui."""

    async def execute(self, actions: Sequence[Action | dict[str, object]]) -> list[ActionResult]:
        return await execute_actions(actions)


__all__ = ["Actuator", "FrameSource", "GameActuator", "GameSource", "Sighting"]
