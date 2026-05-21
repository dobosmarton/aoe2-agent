"""In-process broker for runs currently being generated (Phase 9).

The arena/web/server.py `/events` endpoint reads finalized DuckDB files
in 7.1. Phase 9 adds *live* runs created by `POST /forks` — these need
events to flow from the background `synth_game_loop` writer to SSE
subscribers without DuckDB polling.

`LiveRunRegistry` is a pub/sub broker keyed by `run_id`. Background
fork tasks publish events into it via `_BroadcastingSink`, which also
tees to a `DuckDBEventSink` for durability. SSE handlers subscribe to
the registry while the run is live and fall back to DuckDB-only reads
after `finalize(run_id)`.

Single-event-loop design: every method is synchronous because the
FastAPI loop never preempts non-await statements. Subscribers do their
`await` on `asyncio.Queue.get()`, not on the registry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.event_log import DuckDBEventSink, Event


@dataclass(frozen=True, slots=True)
class _Subscription:
    """One SSE client's listening queue. `None` sentinel = end-of-stream."""

    queue: asyncio.Queue[Event | None]


class LiveRunRegistry:
    """Tracks in-flight runs and fans events out to SSE subscribers."""

    def __init__(self) -> None:
        self._subs: dict[str, list[_Subscription]] = {}

    def is_live(self, run_id: str) -> bool:
        return run_id in self._subs

    def register(self, run_id: str) -> None:
        if run_id in self._subs:
            raise ValueError(f"run_id {run_id!r} is already registered")
        self._subs[run_id] = []

    def subscribe(self, run_id: str) -> _Subscription:
        subs = self._subs.get(run_id)
        if subs is None:
            raise KeyError(f"run_id {run_id!r} is not live")
        sub = _Subscription(queue=asyncio.Queue())
        subs.append(sub)
        return sub

    def unsubscribe(self, run_id: str, sub: _Subscription) -> None:
        subs = self._subs.get(run_id)
        if subs is None:
            return
        try:
            subs.remove(sub)
        except ValueError:
            return

    def publish_nowait(self, event: Event) -> None:
        subs = self._subs.get(event.run_id)
        if subs is None:
            return
        for sub in subs:
            sub.queue.put_nowait(event)

    def finalize(self, run_id: str) -> None:
        subs = self._subs.pop(run_id, None)
        if subs is None:
            return
        for sub in subs:
            sub.queue.put_nowait(None)


@dataclass(frozen=True, slots=True)
class BroadcastingSink:
    """EventSink that fans out to a DuckDB writer AND the live registry.

    Frozen — both deps are required at construction. emit() is synchronous
    because `EventSink.emit` is sync; the registry call uses
    call_soon_threadsafe defensively so a future emit from a worker
    thread still reaches the FastAPI loop safely.
    """

    db_sink: DuckDBEventSink
    registry: LiveRunRegistry
    loop: asyncio.AbstractEventLoop

    def emit(self, event: Event) -> None:
        self.db_sink.emit(event)
        self.loop.call_soon_threadsafe(self.registry.publish_nowait, event)


__all__ = ["BroadcastingSink", "LiveRunRegistry", "_Subscription"]
