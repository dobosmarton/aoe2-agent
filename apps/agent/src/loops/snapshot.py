"""What the three clocks hand each other: one frame, and the pipe it travels.

One thread, so no lock. The discipline that replaces one: the writer swaps a
whole new frame in, so a reader never sees a half-built one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from ..entity_snapshot import snapshot_entities
from ..resource_ocr import ResourceReadings
from ..turn_timing import elapsed_ms

if TYPE_CHECKING:
    from ..entity_snapshot import EntitySnapshot
    from ..policy.state import PolicyState


def _immutable_readings(readings: ResourceReadings) -> ResourceReadings:
    """Detach a HUD publication from the producer's mutable dictionary."""
    return cast("ResourceReadings", MappingProxyType(dict(readings)))


@dataclass(frozen=True, slots=True)
class Perception:
    """One frame, shared by reference across the clocks.

    `captured_at` and `age_ms` mirror `policy.state.PolicyState`, so a frame and
    the state built from it answer the freshness question the same way.
    """

    screenshot: bytes = b""
    width: int = 0
    height: int = 0
    # Accept detector objects or replay mappings at the boundary; __post_init__
    # freezes them all into EntitySnapshot values before publication.
    entities: tuple[object, ...] = ()
    entity_summary: str = ""
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    world: PolicyState | None = None
    ownership: tuple[tuple[str, str, float], ...] = ()
    input_revision: int = 0
    spatial_valid: bool = True
    hud_only: bool = False
    alarm: bool = False
    tick: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # TypedDict describes the OCR keys; copy it so a published frame cannot
        # change when the OCR producer reuses its mutable dictionary.
        object.__setattr__(self, "hud_readings", _immutable_readings(self.hud_readings))
        object.__setattr__(self, "entities", snapshot_entities(self.entities))

    @property
    def age_ms(self) -> float:
        """Milliseconds since this frame was captured."""
        return elapsed_ms(self.captured_at)


@dataclass(frozen=True, slots=True)
class SpatialRefresh:
    """A post-input view for HUD, selection, and coordinate verification."""

    captured_at: float
    input_revision: int
    spatial_valid: bool
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    selected_unit: str | None = None
    screenshot: bytes = b""
    entities: tuple[EntitySnapshot, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hud_readings", _immutable_readings(self.hud_readings))


class FramePipe:
    """The channel from the perceive loop to the other two.

    `latest` never blocks, keeping the act loop off the perception path. `after`
    is the one place a reader waits, and only ONE may: it clears the arrival flag.
    """

    __slots__ = (
        "_arrived",
        "_frame",
        "_selection_only",
        "_spatial_arrived",
        "_spatial_request",
        "_spatial_view",
        "_urgent",
    )

    def __init__(self) -> None:
        self._frame: Perception | None = None
        self._spatial_view: SpatialRefresh | None = None
        self._arrived = asyncio.Event()
        self._spatial_request: asyncio.Future[SpatialRefresh] | None = None
        self._selection_only = False
        self._spatial_arrived = asyncio.Event()
        self._urgent = asyncio.Event()

    def put(self, frame: Perception) -> None:
        """Publish a frame. Perceive only."""
        self._frame = frame
        self._arrived.set()

    def latest(self) -> Perception | None:
        """The newest frame, or None before the first one. Never blocks."""
        return self._frame

    def evidence_screenshot(self) -> bytes:
        """The newest captured view, including a post-navigation refresh."""
        frame = self._frame
        spatial = self._spatial_view
        if spatial is not None and (frame is None or spatial.captured_at >= frame.captured_at):
            return spatial.screenshot
        return frame.screenshot if frame is not None else b""

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

    def request_spatial_refresh(
        self, *, selection_only: bool = False
    ) -> asyncio.Future[SpatialRefresh]:
        """Ask perception for a new view; input ownership allows one waiter."""
        if self.pending_spatial_refresh() is not None:
            raise RuntimeError("a spatial refresh is already pending")
        request: asyncio.Future[SpatialRefresh] = asyncio.get_running_loop().create_future()
        self._spatial_request = request
        self._selection_only = selection_only
        self._spatial_arrived.set()
        self.request_now()
        return request

    def selection_only(self, request: asyncio.Future[SpatialRefresh]) -> bool:
        """Whether this exact pending request needs HUD/selection, not detection."""
        return self._spatial_request is request and self._selection_only

    async def wait_for_spatial_request(self) -> asyncio.Future[SpatialRefresh]:
        """Wake perception immediately when an input-dependent view is needed."""
        while True:
            request = self.pending_spatial_refresh()
            if request is not None:
                return request
            self._spatial_arrived.clear()
            await self._spatial_arrived.wait()

    def pending_spatial_refresh(self) -> asyncio.Future[SpatialRefresh] | None:
        """The current unresolved refresh request, if any."""
        request = self._spatial_request
        return request if request is not None and not request.done() else None

    def complete_spatial_refresh(
        self, request: asyncio.Future[SpatialRefresh], refresh: SpatialRefresh
    ) -> None:
        """Only the requested capture may release its waiter."""
        if self._spatial_request is request and not request.done():
            self._spatial_view = refresh
            request.set_result(refresh)

    def clear_spatial_refresh(self, request: asyncio.Future[SpatialRefresh]) -> None:
        """Forget a completed or timed-out request without touching a newer one."""
        if self._spatial_request is request:
            self._spatial_request = None
            self._selection_only = False

    async def wait_for_due(self, interval: float) -> None:
        """Hold the perceive cadence, cut short by a `request_now`."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._urgent.wait(), timeout=interval)
        self._urgent.clear()


__all__ = ["FramePipe", "Perception", "SpatialRefresh"]
