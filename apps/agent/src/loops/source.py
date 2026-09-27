"""Where a frame comes from, and where an action goes.

Two seams, so the clocks are not welded to one environment: the real game behind
them here, `world_sim` behind them in plan 5.3, lists behind them in a test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, TypeAlias

import structlog

from ..config import config
from ..detection_phase import detect_frame, summarize_frame
from ..executor import execute_actions
from ..providers.strategist import read_hud_readings
from ..screen import capture_screenshot, save_screenshot
from ..window import get_game_window_rect
from .snapshot import FramePipe, Perception

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from detection.inference.detector import EntityDetector
    from detection.inference.frame_diff import FrameDiffer
    from detection.inference.ownership import Owner
    from detection.inference.remote_detector import RemoteDetector

    from ..entity_snapshot import EntitySnapshot
    from ..executor import ActionLedger, ActionResult
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
# How long the executor's rescan hook waits for a fresh frame before giving up.
_REFRESH_TIMEOUT = 3.0


def frame_refresh(frames: FramePipe) -> Callable[[], Awaitable[bool]]:
    """The executor's rescan hook, rerouted to the perceive loop.

    Composite handlers rescan from inside `execute_action`, so the hook is the
    only place that covers every path. A timeout makes the dependent action fail.
    """

    async def refresh() -> bool:
        asked = time.monotonic()
        frames.request_now()
        try:
            await asyncio.wait_for(frames.after(asked), timeout=_REFRESH_TIMEOUT)
            return True
        except TimeoutError:
            log.warning("frame_refresh_timed_out", seconds=_REFRESH_TIMEOUT)
            return False

    return refresh


@dataclass(frozen=True, slots=True)
class Sighting:
    """One perception pass. Ownership rides alongside the frame because only the
    alarm check reads it, and re-classifying cost 15 s (run 2026-08-20)."""

    frame: Perception
    ownership: Mapping[str, tuple[Owner, float]] = field(default_factory=dict)


class FrameSource(Protocol):
    """Perception, whatever is behind it."""

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        """One frame. Records its own `capture`/`ocr`/`detect` phases."""
        ...

    def close(self) -> None:
        """Release whatever the source owns. Never raises."""
        ...


class Actuator(Protocol):
    """Action, whatever is behind it."""

    async def execute(self, actions: Sequence[Action | dict[str, object]]) -> list[ActionResult]:
        """Run the actions in order and report what each one did."""
        ...


def _grab() -> tuple[bytes, int, int, float]:
    """Screenshot plus its capture instant, off the event loop. The stamp comes
    first, so a frame reads older than it is — staleness then errs to skipping."""
    stamped = time.monotonic()
    screenshot, width, height = capture_screenshot()
    return screenshot, width, height, stamped


class GameSource:
    """The real game: mss, YOLO and local OCR."""

    def __init__(
        self,
        detector: Detector | None = None,
        overlay: DetectionOverlay | None = None,
        frame_differ: FrameDiffer | None = None,
        screenshots_dir: Path | None = None,
        ledger: ActionLedger | None = None,
    ) -> None:
        self._detector = detector
        self._overlay = overlay
        self._differ = frame_differ
        self._screenshots_dir = screenshots_dir
        self._ledger = ledger

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        revision = self._ledger.input_revision if self._ledger is not None else 0
        screenshot, width, height, captured_at = await self._screen(tick, timings)
        hud_readings = await self._hud(screenshot, tick, timings)
        entities, entity_summary, ownership = await self._entities(screenshot, timings)
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

    async def _screen(self, tick: int, timings: TickTimings) -> tuple[bytes, int, int, float]:
        """Grab the frame. The overlay hides first, so it stays out of the shot."""
        with timings.phase("capture"):
            if self._overlay:
                self._overlay.hide()
            screenshot, width, height, captured_at = await asyncio.to_thread(_grab)
            self._save_sample(screenshot, tick)
        return screenshot, width, height, captured_at

    async def _hud(self, screenshot: bytes, tick: int, timings: TickTimings) -> ResourceReadings:
        """Read the resource bar, and show the OCR boxes it calibrated."""
        with timings.phase("ocr"):
            hud_readings, calib = await read_hud_readings(screenshot, turn=tick)
        if self._overlay is not None and calib is not None:
            self._overlay.set_ocr_fields(calib.field_rects())
        return hud_readings

    async def _entities(
        self, screenshot: bytes, timings: TickTimings
    ) -> tuple[list[EntitySnapshot], str, Mapping[str, tuple[Owner, float]]]:
        """Detect, then tag ownership. Empty without a detector."""
        with timings.phase("detect"):
            entities: list[EntitySnapshot] = []
            if self._detector:
                entities = list(await detect_frame(self._detector, self._differ, screenshot))
            if self._overlay is not None:
                self._overlay.show(entities, get_game_window_rect())
            entity_summary, ownership = await summarize_frame(entities, screenshot)
        return entities, entity_summary, ownership

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
