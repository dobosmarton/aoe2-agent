"""What the three clocks hand each other: one frame, and the pipe it travels.

One thread, so no lock. The discipline that replaces one: the writer swaps a
whole new frame in, so a reader never sees a half-built one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from ..resource_ocr import ResourceReadings
from ..turn_timing import elapsed_ms


@dataclass(frozen=True, slots=True)
class Perception:
    """One frame, shared by reference across the three loops.

    `captured_at` and `age_ms` mirror `policy.state.PolicyState`, so a frame and
    the state built from it answer the freshness question the same way.
    """

    screenshot: bytes = b""
    width: int = 0
    height: int = 0
    # `DetectedEntity` from the detector, dicts from a replay or the simulator.
    # `entity_utils.extract_attrs` reads either.
    entities: tuple[object, ...] = ()
    entity_summary: str = ""
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    alarm: bool = False
    tick: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def age_ms(self) -> float:
        """Milliseconds since this frame was captured."""
        return elapsed_ms(self.captured_at)


class FramePipe:
    """The channel from the perceive loop to the other two.

    `latest` never blocks, keeping the act loop off the perception path. `after`
    is the one place a reader waits, and only ONE may: it clears the arrival flag.
    """

    __slots__ = ("_arrived", "_frame", "_urgent")

    def __init__(self) -> None:
        self._frame: Perception | None = None
        self._arrived = asyncio.Event()
        self._urgent = asyncio.Event()

    def put(self, frame: Perception) -> None:
        """Publish a frame. Perceive only."""
        self._frame = frame
        self._arrived.set()

    def latest(self) -> Perception | None:
        """The newest frame, or None before the first one. Never blocks."""
        return self._frame

    async def after(self, captured_at: float) -> Perception:
        """The first frame captured after `captured_at`."""
        while True:
            self._arrived.clear()
            frame = self._frame
            if frame is not None and frame.captured_at > captured_at:
                return frame
            await self._arrived.wait()

    def request_now(self) -> None:
        """Ask the perceive loop to skip the rest of its wait."""
        self._urgent.set()

    async def wait_for_due(self, interval: float) -> None:
        """Hold the perceive cadence, cut short by a `request_now`."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._urgent.wait(), timeout=interval)
        self._urgent.clear()


__all__ = ["FramePipe", "Perception"]
