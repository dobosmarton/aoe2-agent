"""Where a frame comes from, and where an action goes.

Two seams, so the clocks are not welded to one environment: the real game behind
them here, `world_sim` behind them in plan 5.3, lists behind them in a test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from ..config import config
from ..detection_phase import (
    _classify_entities,
    _register_rescan_callbacks,
    _run_detection,
)
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

    from ..executor import ActionResult
    from ..models import Action
    from ..overlay import DetectionOverlay
    from ..turn_timing import TickTimings

    # Mirrors `detection_phase.Detector` — local weights or the remote server.
    Detector = EntityDetector | RemoteDetector

log = structlog.stdlib.get_logger()

# Frames between screenshot saves. Saving every frame was affordable at ~10 s
# per turn; the perceive loop runs far more often, and the write blocks it.
_SCREENSHOT_SAMPLE = 10
# How long the executor's rescan hook waits for a fresh frame before giving up.
_REFRESH_TIMEOUT = 3.0


def frame_refresh(frames: FramePipe) -> Callable[[], Awaitable[None]]:
    """The executor's rescan hook, rerouted to the perceive loop.

    Composite handlers rescan from inside `execute_action`, so the hook is the
    only place that covers every path. A timeout proceeds on the stale frame.
    """

    async def refresh() -> None:
        asked = time.monotonic()
        frames.request_now()
        try:
            await asyncio.wait_for(frames.after(asked), timeout=_REFRESH_TIMEOUT)
        except TimeoutError:
            log.warning("frame_refresh_timed_out", seconds=_REFRESH_TIMEOUT)

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
    ) -> None:
        self._detector = detector
        self._overlay = overlay
        self._screenshots_dir = screenshots_dir
        if detector is not None:
            # For the combat tool loop's mid-turn rescan. The act loop never
            # asks for one (plan 3.5).
            _register_rescan_callbacks(detector, overlay, frame_differ)

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        with timings.phase("capture"):
            if self._overlay:
                self._overlay.hide()
            screenshot, width, height, captured_at = await asyncio.to_thread(_grab)
            self._save_sample(screenshot, tick)

        with timings.phase("ocr"):
            hud_readings, calib = await read_hud_readings(screenshot, turn=tick)
        if self._overlay is not None and calib is not None:
            self._overlay.set_ocr_fields(calib.field_rects())

        with timings.phase("detect"):
            entities: list[object] = []
            if self._detector:
                entities = await _run_detection(self._detector, screenshot, tick, alarm=False)
            if self._overlay is not None:
                self._overlay.show(entities, get_game_window_rect())
            entity_summary, ownership = await _classify_entities(entities, screenshot)

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
            ),
            ownership=ownership,
        )

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
