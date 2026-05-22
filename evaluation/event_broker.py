"""Event broker — Phase 1 of the log-first SSE architecture.

The broker is the single source of truth for *live* event consumers. DuckDB
and SSE clients are symmetric materializations that subscribe via
`broker.stream(...)`. Neither knows about the other.

See `docs/design/event-broker-architecture.md` for the full motivation
(spoiler: today's `/events` handler races with the writer-side replay task
for the same DuckDB file). This module is a pure-domain leaf — it does not
import FastAPI, DuckDB, or anything from `arena/`.

Phase 1 scope:
    - The `EventBroker` Protocol (interface).
    - `InProcessEventBroker` — single-process, asyncio-native implementation.
    - `BrokerEventSink` — adapter that lets a sync `EventSink` caller publish
      onto the broker's loop from any thread.

Phase 2 will wire `arena/web/forks.py` to publish through the broker and
delete the file-coupled live path. Phase C swaps in a Redis/NATS broker
behind the same Protocol — every consumer keeps working.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, NewType, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from evaluation.event_log import Event

# ---------------------------------------------------------------------------
# Identity types — make illegal states unrepresentable at the type-checker.
# ---------------------------------------------------------------------------

RunId = NewType("RunId", str)
"""Stable per-run identifier (UUID hex today; opaque to the broker)."""

Seq = NewType("Seq", int)
"""Monotonic per-run sequence number, starting at 1, assigned by the broker.

Distinct from `Event.t` (which is the *turn* number — multiple events per turn
share the same `t`). `Seq` makes ordering total across the whole run, which
the live consumer needs for unambiguous dedupe on reconnect."""


# ---------------------------------------------------------------------------
# Envelope — what consumers iterate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """One sequenced event delivery. Immutable.

    Wrapping `Event` instead of adding `seq` to the dataclass keeps the
    broker's "I assigned this" concern out of the domain model and means
    no DuckDB schema change.
    """

    run_id: RunId
    seq: Seq
    event: Event


# ---------------------------------------------------------------------------
# Broker Protocol.
# ---------------------------------------------------------------------------


class EventBroker(Protocol):
    """Pub/sub with replay-from-offset semantics.

    Producers call `open_run` -> `publish` * N -> `close_run`.
    Consumers call `stream(run_id, from_seq)` — one call covers replay+tail.
    """

    def open_run(self, run_id: RunId) -> None:
        """Begin a new run. Raises if already open."""
        ...

    def close_run(self, run_id: RunId) -> None:
        """Mark a run finalized. Active streams drain and return."""
        ...

    def is_open(self, run_id: RunId) -> bool:
        """True iff the run is currently accepting publishes."""
        ...

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        """Append `event` to the run's log. Returns the assigned `Seq`.

        Raises `RuntimeError` if the run is not open.
        """
        ...

    def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]:
        """Yield every envelope with `seq >= from_seq`, then tail live ones.

        Terminates when the run is closed AND the buffer is drained.
        Cancelling the consumer cleanly deregisters its waiter.
        """
        ...


# ---------------------------------------------------------------------------
# In-process implementation.
# ---------------------------------------------------------------------------


class InProcessEventBroker:
    """Single-process, asyncio-native broker.

    State model:
        - `_buffers[run]` — append-only list of envelopes, retained for the
          lifetime of the run (and beyond `close_run`, so late subscribers
          can still replay; explicit reap is Phase 3 work).
        - `_seq[run]` — `itertools.count(1)`, source of `Seq` values.
        - `_open` — set of currently-publishing runs.
        - `_waiters[run]` — async `Event`s woken on each publish or close.

    Memory: O(events_per_run * concurrent_runs). For arena replays
    (low-hundreds of events per run), negligible.

    Concurrency: all methods assume a single asyncio event loop. Cross-thread
    publishing is the `BrokerEventSink` adapter's job, not the broker's.
    """

    def __init__(self) -> None:
        self._buffers: dict[RunId, list[EventEnvelope]] = {}
        self._seq: dict[RunId, count[int]] = {}
        self._open: set[RunId] = set()
        self._waiters: dict[RunId, list[asyncio.Event]] = defaultdict(list)

    def open_run(self, run_id: RunId) -> None:
        if run_id in self._open:
            raise ValueError(f"run {run_id!r} is already open")
        self._buffers.setdefault(run_id, [])
        self._seq[run_id] = count(1)
        self._open.add(run_id)

    def close_run(self, run_id: RunId) -> None:
        self._open.discard(run_id)
        # Wake any in-flight streams so they observe the closed state and exit.
        for waiter in self._waiters.pop(run_id, ()):
            waiter.set()

    def is_open(self, run_id: RunId) -> bool:
        return run_id in self._open

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        if run_id not in self._open:
            raise RuntimeError(f"run {run_id!r} is not open")
        if event.run_id != run_id:
            # Catches accidental swaps at the producer boundary. The broker
            # uses run_id for routing; event.run_id is what consumers read.
            # Drifting these silently corrupts downstream materializations.
            raise ValueError(
                f"event.run_id {event.run_id!r} does not match broker run_id {run_id!r}"
            )
        seq = Seq(next(self._seq[run_id]))
        self._buffers[run_id].append(EventEnvelope(run_id=run_id, seq=seq, event=event))
        for waiter in self._waiters[run_id]:
            waiter.set()
        return seq

    async def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]:
        # `Seq(0)` means "from the beginning"; otherwise skip everything below.
        cursor = max(0, int(from_seq) - 1)
        while True:
            buf = self._buffers.get(run_id, [])
            while cursor < len(buf):
                yield buf[cursor]
                cursor += 1
            if not self.is_open(run_id):
                return
            # Arm a waiter, then re-check the buffer length on wake — publish
            # appends *before* signalling, so this loop never misses an event.
            waiter = asyncio.Event()
            self._waiters[run_id].append(waiter)
            try:
                await waiter.wait()
            finally:
                # Remove without mutating-while-iterating in publish/close —
                # filter-rebind is safe because publish does `for w in list`
                # over the current binding and we replace the binding here.
                self._waiters[run_id] = [w for w in self._waiters[run_id] if w is not waiter]


# ---------------------------------------------------------------------------
# Producer adapter — bridges sync EventSink callers to async publish().
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrokerEventSink:
    """`EventSink` adapter that publishes onto a broker's event loop.

    The existing producer API (`evaluation.event_log.EventSink`) is sync —
    `emit(event) -> None`. The broker's `publish` is async because future
    implementations (Redis, NATS) will do network I/O. Bridging via
    `loop.call_soon_threadsafe(asyncio.create_task, broker.publish(...))`
    keeps `emit` non-blocking and preserves FIFO order per loop thread.
    """

    broker: EventBroker
    run_id: RunId
    loop: asyncio.AbstractEventLoop

    def emit(self, event: Event) -> None:
        # `broker.publish(...)` returns a coroutine eagerly in the caller's
        # thread; `create_task` binds it to the broker's loop on wake-up.
        self.loop.call_soon_threadsafe(
            asyncio.create_task,
            self.broker.publish(self.run_id, event),
        )


__all__ = [
    "BrokerEventSink",
    "EventBroker",
    "EventEnvelope",
    "InProcessEventBroker",
    "RunId",
    "Seq",
]
