"""What the three clocks hand each other: one frame, and the pipe it travels.

One thread, so no lock. The discipline that replaces one: the writer swaps a
whole new frame in, so a reader never sees a half-built one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from ..entity_snapshot import snapshot_entities
from ..resource_ocr import ResourceReadings
from ..turn_timing import elapsed_ms

if TYPE_CHECKING:
    from ..entity_snapshot import EntitySnapshot
    from ..policy.state import PolicyState
    from .source import Sighting


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
    window_rect: tuple[int, int, int, int] | None = None
    # Accept detector objects or replay mappings at the boundary; __post_init__
    # freezes them all into EntitySnapshot values before publication.
    entities: tuple[object, ...] = ()
    entity_summary: str = ""
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    world: PolicyState | None = None
    ownership: tuple[tuple[str, str, float], ...] = ()
    input_revision: int = 0
    camera_revision: int = 0
    economic_revision: int = 0
    selection_revision: int = 0
    capture_id: int = -1
    spatial_capture_id: int = -1
    hud_capture_id: int = -1
    publication_version: int = 0
    spatial_version: int = 0
    hud_version: int = 0
    spatial_captured_at: float | None = None
    hud_captured_at: float | None = None
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
    sighting: Sighting | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hud_readings", _immutable_readings(self.hud_readings))


class FramePipe:
    """The channel from the perceive loop to the other two.

    `latest` never blocks. Independent actor and deliberate readers are both
    woken for each publication; neither can consume the other's notification.
    """

    __slots__ = (
        "_frame",
        "_hud_version",
        "_publication_version",
        "_selection_only",
        "_spatial_arrived",
        "_spatial_request",
        "_spatial_version",
        "_spatial_view",
        "_urgent",
        "_waiters",
    )

    def __init__(self) -> None:
        self._frame: Perception | None = None
        self._spatial_view: SpatialRefresh | None = None
        self._waiters: set[asyncio.Future[None]] = set()
        self._spatial_version = 0
        self._hud_version = 0
        self._publication_version = 0
        self._spatial_request: asyncio.Future[SpatialRefresh] | None = None
        self._selection_only = False
        self._spatial_arrived = asyncio.Event()
        self._urgent = asyncio.Event()

    def put(self, frame: Perception) -> Perception:
        """Publish a frame. Perceive only."""
        self._publication_version += 1
        published = replace(frame, publication_version=self._publication_version)
        self._frame = published
        for waiter in tuple(self._waiters):
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()
        return published

    def put_hud(self, frame: Perception) -> Perception:
        """Merge a HUD update without making an older spatial view look fresh."""
        previous = self._frame
        if (
            previous is not None
            and previous.hud_captured_at is not None
            and frame.captured_at <= previous.hud_captured_at
        ):
            return previous
        self._hud_version += 1
        if previous is None:
            published = replace(frame, hud_version=self._hud_version)
            return self.put(published)
        world = frame.world
        if world is not None and previous.world is not None:
            world = replace(
                world,
                visible_classes=previous.world.visible_classes,
                own_army_present=previous.world.own_army_present,
                spatial_valid=(
                    previous.world.spatial_valid
                    and previous.camera_revision == frame.camera_revision
                ),
            )
        combined = Perception(
            screenshot=previous.screenshot,
            width=previous.width,
            height=previous.height,
            window_rect=previous.window_rect,
            entities=previous.entities,
            entity_summary=previous.entity_summary,
            hud_readings=frame.hud_readings,
            world=world,
            ownership=previous.ownership,
            input_revision=frame.input_revision,
            camera_revision=previous.camera_revision,
            economic_revision=frame.economic_revision,
            selection_revision=previous.selection_revision,
            capture_id=frame.capture_id,
            spatial_capture_id=previous.spatial_capture_id,
            hud_capture_id=frame.hud_capture_id,
            spatial_version=previous.spatial_version,
            hud_version=self._hud_version,
            spatial_captured_at=previous.spatial_captured_at,
            hud_captured_at=frame.captured_at,
            spatial_valid=(
                previous.spatial_valid and previous.camera_revision == frame.camera_revision
            ),
            hud_only=True,
            alarm=previous.alarm,
            tick=frame.tick,
            captured_at=max(frame.captured_at, previous.captured_at),
        )
        return self.put(combined)

    def put_spatial(self, frame: Perception) -> Perception:
        """Publish fresh boxes alongside the last compatible HUD reading."""
        self._spatial_version += 1
        previous = self._frame
        if (
            frame.hud_captured_at is not None
            and frame.hud_readings
            and (
                previous is None
                or previous.hud_captured_at is None
                or frame.hud_captured_at >= previous.hud_captured_at
            )
        ):
            self._hud_version += 1
            return self.put(
                replace(
                    frame,
                    spatial_version=self._spatial_version,
                    hud_version=self._hud_version,
                )
            )
        if (
            previous is None
            or previous.hud_captured_at is None
            or previous.economic_revision != frame.economic_revision
        ):
            published = replace(frame, spatial_version=self._spatial_version)
            return self.put(published)
        world = frame.world
        if world is not None and previous.world is not None:
            world = replace(
                world,
                food=previous.world.food,
                wood=previous.world.wood,
                gold=previous.world.gold,
                stone=previous.world.stone,
                population=previous.world.population,
                population_cap=previous.world.population_cap,
                age=previous.world.age,
                idle_count=previous.world.idle_count,
                villager_jobs=previous.world.villager_jobs,
                captured_at=previous.world.captured_at,
            )
        combined = Perception(
            screenshot=frame.screenshot,
            width=frame.width,
            height=frame.height,
            window_rect=frame.window_rect,
            entities=frame.entities,
            entity_summary=frame.entity_summary,
            hud_readings=previous.hud_readings,
            world=world,
            ownership=frame.ownership,
            input_revision=frame.input_revision,
            camera_revision=frame.camera_revision,
            economic_revision=previous.economic_revision,
            selection_revision=frame.selection_revision,
            capture_id=frame.capture_id,
            spatial_capture_id=frame.spatial_capture_id,
            hud_capture_id=previous.hud_capture_id,
            spatial_version=self._spatial_version,
            hud_version=previous.hud_version,
            spatial_captured_at=frame.spatial_captured_at,
            hud_captured_at=previous.hud_captured_at,
            spatial_valid=frame.spatial_valid,
            hud_only=False,
            alarm=frame.alarm,
            tick=frame.tick,
            captured_at=max(frame.captured_at, previous.captured_at),
        )
        return self.put(combined)

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
            frame = self._frame
            if frame is not None and frame.captured_at > captured_at:
                return frame
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.discard(waiter)

    async def after_version(self, publication_version: int) -> Perception:
        """Wait for a new publication, even when its capture predates the HUD."""
        while True:
            frame = self._frame
            if frame is not None and frame.publication_version > publication_version:
                return frame
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.discard(waiter)

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
