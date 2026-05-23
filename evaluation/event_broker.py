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

Phase 3 adds operational hardening on the in-process impl: bounded per-run
buffers (deque with `maxlen`), explicit `reap(run_id)` for memory
reclamation, and `metrics()` for a `/metrics` endpoint. Backpressure
overflow is signalled by raising `BrokerOverflowError` from inside
`stream()` — the slow consumer self-evicts; publishers never block or
raise on buffer pressure.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
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
# Backpressure — signalled by raising from inside `stream()`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrokerOverflowError(Exception):
    """Raised inside `stream()` when the consumer's cursor was evicted.

    `requested_seq` is the next seq the consumer was about to yield;
    `available_from` is the lowest seq still in the broker's buffer
    (a.k.a. `head_seq`). A client recovers by reconnecting with
    `from_seq=available_from` and accepting that events in
    `[requested_seq, available_from)` are lost — surfacing this rather
    than silently dropping is the design's strongest guarantee.
    """

    run_id: RunId
    requested_seq: Seq
    available_from: Seq


# ---------------------------------------------------------------------------
# Metrics — operational snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrokerMetricsSnapshot:
    """Point-in-time counters for `/metrics`. Frozen so callers can't
    mutate broker state through the returned object."""

    events_published: int
    events_streamed: int
    streams_dropped: int
    runs_open: int

    def to_dict(self) -> dict[str, int]:
        """Explicit construction (not `dataclasses.asdict`) keeps the return
        type as `dict[str, int]` instead of `dict[str, Any]` — `asdict`
        recursively normalizes nested values and erases their static type.
        The dataclass field list is the single source of truth; consumers
        (FastAPI `/metrics`, future stats endpoints) get a typed dict
        without restating field names."""
        return {
            "events_published": self.events_published,
            "events_streamed": self.events_streamed,
            "streams_dropped": self.streams_dropped,
            "runs_open": self.runs_open,
        }


# ---------------------------------------------------------------------------
# Broker Protocol.
# ---------------------------------------------------------------------------


class EventBroker(Protocol):
    """Pub/sub with replay-from-offset semantics.

    Producers call `open_run` -> `publish` * N -> `close_run` -> `reap`.
    Consumers call `stream(run_id, from_seq)` — one call covers replay+tail.
    """

    def open_run(self, run_id: RunId) -> None:
        """Begin a new run. Raises if already open."""
        ...

    def close_run(self, run_id: RunId) -> None:
        """Mark a run finalized. Active streams drain and return."""
        ...

    def reap(self, run_id: RunId) -> None:
        """Drop all retained state for a closed run.

        Raises `ValueError` if the run is still open — `reap` after
        `close_run` is the lifecycle order; calling it on an open run is
        an ordering bug, not a runtime condition to silently handle.
        """
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
        Raises `BrokerOverflowError` if `from_seq` (or a cursor that
        falls behind mid-stream) is below the buffer's `head_seq` —
        the slow consumer self-evicts. Cancelling the consumer cleanly
        deregisters its waiter.
        """
        ...


# ---------------------------------------------------------------------------
# In-process implementation.
# ---------------------------------------------------------------------------


_DEFAULT_MAX_BUFFER_SIZE = 10_000


class InProcessEventBroker:
    """Single-process, asyncio-native broker.

    State model:
        - `_buffers[run]` — bounded `deque` (per-run ring buffer). On
          overflow, the oldest envelope is auto-evicted by deque (O(1))
          and `_head_seq[run]` advances by one.
        - `_head_seq[run]` — lowest `Seq` still in `_buffers[run]`.
          Starts at 1 on `open_run`; bumped immediately before each
          eviction-triggering append.
        - `_seq[run]` — `itertools.count(1)`, source of `Seq` values.
        - `_open` — set of currently-publishing runs.
        - `_waiters[run]` — async `Event`s woken on each publish or close.

    Memory: O(min(events_per_run, max_buffer_size) * concurrent_runs).
    Explicit `reap(run_id)` after `close_run` drops everything for a run.

    Concurrency: all methods assume a single asyncio event loop. The
    `stream()` overflow-check correctness rests on this — `publish()`
    runs to completion between consumer `await`s, so a suspended
    consumer's cursor cannot be evicted mid-flight; only on wake does
    it re-read `head_seq` and self-raise. Cross-thread publishing is
    the `BrokerEventSink` adapter's job; moving the broker itself to
    a thread pool would invalidate this design (Phase C swap must
    either preserve the invariant or replace the eviction strategy).

    Cross-thread publishing is the `BrokerEventSink` adapter's job, not
    the broker's.
    """

    def __init__(self, *, max_buffer_size: int = _DEFAULT_MAX_BUFFER_SIZE) -> None:
        self._max_buffer_size = max_buffer_size
        self._buffers: dict[RunId, deque[EventEnvelope]] = {}
        self._head_seq: dict[RunId, int] = {}
        self._seq: dict[RunId, count[int]] = {}
        self._open: set[RunId] = set()
        self._waiters: dict[RunId, list[asyncio.Event]] = defaultdict(list)
        self._metrics_events_published = 0
        self._metrics_events_streamed = 0
        self._metrics_streams_dropped = 0

    def open_run(self, run_id: RunId) -> None:
        if run_id in self._open:
            raise ValueError(f"run {run_id!r} is already open")
        self._buffers.setdefault(run_id, deque(maxlen=self._max_buffer_size))
        self._head_seq.setdefault(run_id, 1)
        self._seq[run_id] = count(1)
        self._open.add(run_id)

    def close_run(self, run_id: RunId) -> None:
        self._open.discard(run_id)
        # Wake any in-flight streams so they observe the closed state and exit.
        for waiter in self._waiters.pop(run_id, ()):
            waiter.set()

    def reap(self, run_id: RunId) -> None:
        if run_id in self._open:
            raise ValueError(f"cannot reap open run {run_id!r}; close it first")
        self._buffers.pop(run_id, None)
        self._head_seq.pop(run_id, None)
        self._seq.pop(run_id, None)

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
        buf = self._buffers[run_id]
        # When the deque is at maxlen, the next append evicts the leftmost
        # element; bump head_seq BEFORE the append so the invariant
        # "buf[0].seq == head_seq" holds at every observable moment.
        if len(buf) >= self._max_buffer_size:
            self._head_seq[run_id] += 1
        seq = Seq(next(self._seq[run_id]))
        buf.append(EventEnvelope(run_id=run_id, seq=seq, event=event))
        self._metrics_events_published += 1
        for waiter in self._waiters[run_id]:
            waiter.set()
        return seq

    async def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]:
        # `cursor` tracks the seq of the last envelope yielded (or 0
        # initially). `from_seq=Seq(0)` means "from the beginning".
        cursor = max(0, int(from_seq) - 1)
        while True:
            head = self._head_seq.get(run_id, 1)
            # Eviction check at the outer-loop top, NOT mid-yield: the
            # single-loop-thread invariant means publish can't run while
            # this generator is between yields, so the cursor only becomes
            # invalid across an `await waiter.wait()`. Checking on wake
            # is sufficient.
            if cursor + 1 < head:
                self._metrics_streams_dropped += 1
                raise BrokerOverflowError(
                    run_id=run_id,
                    requested_seq=Seq(cursor + 1),
                    available_from=Seq(head),
                )
            buf = self._buffers.get(run_id)
            if buf is not None:
                local_offset = (cursor + 1) - head
                while local_offset < len(buf):
                    yield buf[local_offset]
                    cursor += 1
                    local_offset += 1
                    self._metrics_events_streamed += 1
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

    def metrics(self) -> BrokerMetricsSnapshot:
        """Snapshot the four operational counters.

        Not on the `EventBroker` Protocol — counter exposure is impl-specific.
        Future Redis/NATS brokers will expose stats via their own surfaces
        (Redis `INFO`, NATS monitoring port), so the FastAPI `/metrics`
        endpoint reaches through with an `isinstance`/`cast` rather than
        polymorphically.
        """
        return BrokerMetricsSnapshot(
            events_published=self._metrics_events_published,
            events_streamed=self._metrics_events_streamed,
            streams_dropped=self._metrics_streams_dropped,
            runs_open=len(self._open),
        )


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
    "BrokerMetricsSnapshot",
    "BrokerOverflowError",
    "EventBroker",
    "EventEnvelope",
    "InProcessEventBroker",
    "RunId",
    "Seq",
]
