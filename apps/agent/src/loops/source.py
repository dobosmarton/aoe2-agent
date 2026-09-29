"""Where a frame comes from, and where an action goes.

Two seams, so the clocks are not welded to one environment: the real game behind
them here, `world_sim` behind them in plan 5.3, lists behind them in a test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

import structlog

from ..config import config
from ..detection_phase import detect_frame, summarize_frame
from ..executor import execute_actions
from ..providers.strategist import read_hud_readings
from ..resource_ocr import calibration_for, read_selected_unit
from ..screen import capture_screenshot_with_native_hud, save_screenshot
from ..window import get_game_window_rect
from .snapshot import FramePipe, Perception, SpatialRefresh

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from detection.inference.detector import EntityDetector
    from detection.inference.frame_diff import FrameDiffer
    from detection.inference.ownership import Owner
    from detection.inference.remote_detector import RemoteDetector

    from ..entity_snapshot import EntitySnapshot
    from ..executor import ActionLedger, ActionResult
    from ..game_profile import GameProfile
    from ..models import Action
    from ..overlay import DetectionOverlay
    from ..resource_ocr import ResourceReadings
    from ..turn_timing import TickTimings

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
        frame_differ: FrameDiffer | None = None,
        screenshots_dir: Path | None = None,
        ledger: ActionLedger | None = None,
        profile: GameProfile | None = None,
    ) -> None:
        self._detector = detector
        self._overlay = overlay
        self._differ = frame_differ
        self._screenshots_dir = screenshots_dir
        self._ledger = ledger
        self._profile = profile

    async def capture_hud(self, tick: int, timings: TickTimings) -> Sighting | None:
        """Publish economic facts before the more expensive full detector pass."""
        revision = self._ledger.input_revision if self._ledger is not None else 0
        screenshot, native_hud, width, height, captured_at = await self._screen(None, timings)
        hud_readings = await self._hud(native_hud, tick, timings, (width, height))
        valid = self._ledger is None or self._ledger.input_revision == revision
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
                spatial_valid=False,
                hud_only=True,
            )
        )

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        revision = self._ledger.input_revision if self._ledger is not None else 0
        screenshot, native_hud, width, height, captured_at = await self._screen(tick, timings)
        hud_readings, entity_result = await asyncio.gather(
            self._hud(native_hud, tick, timings, (width, height)),
            self._entities(screenshot, timings),
        )
        entities, entity_summary, ownership = entity_result
        spatial_valid = self._ledger is None or self._ledger.input_revision == revision
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
                spatial_valid=spatial_valid,
            ),
            ownership=ownership,
        )

    async def capture_spatial(
        self, timings: TickTimings, *, selection_only: bool = False
    ) -> SpatialRefresh:
        """Refresh coordinates, HUD baseline, and selected command-panel unit."""
        revision = self._ledger.input_revision if self._ledger is not None else 0
        screenshot, native_hud, _width, _height, captured_at = await self._screen(None, timings)
        selection_calibration = calibration_for(_width, _height)

        async def selected() -> str | None:
            if selection_calibration is None:
                return None
            return await asyncio.to_thread(read_selected_unit, screenshot, selection_calibration)

        async def detect() -> list[EntitySnapshot]:
            with timings.phase("detect"):
                return await self._detect_entities(screenshot, fresh=True)

        if selection_only:
            hud_readings, selected_unit = await asyncio.gather(
                self._hud(native_hud, 0, timings, (_width, _height)), selected()
            )
            entities: list[EntitySnapshot] = []
        else:
            hud_readings, selected_unit, entities = await asyncio.gather(
                self._hud(native_hud, 0, timings, (_width, _height)), selected(), detect()
            )
        spatial_valid = self._ledger is None or self._ledger.input_revision == revision
        log.info(
            "spatial_refresh_captured",
            entity_count=len(entities),
            input_revision=revision,
            spatial_valid=spatial_valid,
            selected_unit=selected_unit,
            selection_only=selection_only,
        )
        return SpatialRefresh(
            captured_at=captured_at,
            input_revision=revision,
            spatial_valid=spatial_valid,
            hud_readings=hud_readings,
            selected_unit=selected_unit,
            screenshot=screenshot,
            entities=tuple(entities) if not selection_only else None,
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

    async def _detect_entities(
        self, screenshot: bytes, *, fresh: bool = False
    ) -> list[EntitySnapshot]:
        entities: list[EntitySnapshot] = []
        if self._detector:
            entities = list(
                await detect_frame(
                    self._detector,
                    None if fresh else self._differ,
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
        save_screenshot(screenshot, str(self._screenshots_dir / f"{stamp}_{tick:05d}.jpg"))

    def close(self) -> None:
        if self._overlay:
            self._overlay.close()


class GameActuator:
    """The real game: synthetic mouse and keyboard through pyautogui."""

    async def execute(self, actions: Sequence[Action | dict[str, object]]) -> list[ActionResult]:
        return await execute_actions(actions)


__all__ = ["Actuator", "FrameSource", "GameActuator", "GameSource", "Sighting"]
