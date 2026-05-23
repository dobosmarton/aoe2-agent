"""DuckDB persister — one of N event-broker consumers.

This is the structural inversion that makes the architecture work:
nobody but the persister opens the writer's DuckDB file. The SSE handler
reads from the broker for live runs; only the cold path (after
`broker.close_run`) touches DuckDB read-only. The in-process file-coupling
bug disappears as a side effect.

Two flavors:

  * `persist_to_duckdb(broker, run_id, db_path)` — fork case. Exactly
    one run_id; the persister owns the DuckDB connection for its
    lifetime. One file per run.

  * `MultiRunBrokerSink` + `persist_to_duckdb_via_sink` — CLI case
    (Phase 2.5). Many run_ids per CLI command (e.g. one game loop per
    profile in `race`), but a single DuckDB file. The CLI owns the
    connection; the multi-run sink auto-opens broker runs as it sees
    new run_ids in the event stream and spawns a per-run drainer that
    writes through the shared `DuckDBEventSink`.

Usage — fork case (single run, single file):

    broker.open_run(run_id)
    asyncio.create_task(persist_to_duckdb(broker, run_id, db_path))
    # ... producer publishes ...
    broker.close_run(run_id)   # persister drains buffer and returns

Usage — CLI case (many runs, single file):

    broker = InProcessEventBroker()
    with duckdb.connect(db_path) as conn:
        db_sink = DuckDBEventSink(conn)
        sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop())
        try:
            await producer(sink=sink)
        finally:
            await sink.close_all()  # closes every opened run, awaits persisters
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import duckdb

from evaluation.event_broker import RunId, Seq
from evaluation.event_log import DuckDBEventSink

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.event_broker import EventBroker
    from evaluation.event_log import Event


async def persist_to_duckdb(
    broker: EventBroker,
    run_id: RunId,
    db_path: Path,
) -> None:
    """Drain every published event for `run_id` into a DuckDB file.

    Owns the DuckDB connection exclusively for this run — no other task
    or process should open `db_path` while this coroutine is running.

    Subscribes from `Seq(0)` (whole history) and returns when the broker
    closes the run and the local buffer is drained.
    """
    with duckdb.connect(str(db_path)) as conn:
        sink = DuckDBEventSink(conn)
        async for envelope in broker.stream(run_id, from_seq=Seq(0)):
            sink.emit(envelope.event)


async def persist_to_duckdb_via_sink(
    broker: EventBroker,
    run_id: RunId,
    sink: DuckDBEventSink,
) -> None:
    """Drain `run_id`'s broker stream into a caller-owned DuckDB sink.

    Unlike `persist_to_duckdb`, the caller owns the connection lifecycle.
    Use this when multiple concurrent runs share one DuckDB file in a
    single asyncio context — CLI commands that write many run_ids to
    one log file (`arena race`, `arena rank`) are the canonical case.

    Multiple coroutines may invoke this concurrently against the same
    `sink`; `DuckDBEventSink.emit` is a single synchronous INSERT and
    they serialize naturally on the loop thread.
    """
    async for envelope in broker.stream(run_id, from_seq=Seq(0)):
        sink.emit(envelope.event)


@dataclass(slots=True)
class MultiRunBrokerSink:
    """`EventSink` that routes per-run events onto a shared broker.

    Auto-opens a broker run on first sight of each `event.run_id` and
    spawns a `persist_to_duckdb_via_sink` task to drain that run into
    `db_sink`. Subsequent events for the same run_id publish onto the
    already-open run.

    Why this shape:
        * Fork producers know their run_id up front and use
          `BrokerEventSink` directly. CLI producers (`race`, `rank`)
          generate run_ids inside `synth_game_loop` — they only surface
          at emit time. This sink absorbs that mismatch without forcing
          every game loop API to accept pre-allocated run_ids.
        * One DuckDB file per CLI command is the existing operator
          invariant (multiple run_ids share the `events` table via the
          `run_id` column). A single shared `DuckDBEventSink` preserves
          that property; per-run persister coroutines drain in parallel
          but emit serially on the loop thread.

    Lifecycle ordering inside `_handle_emit` is load-bearing: open the
    run BEFORE spawning the persister BEFORE publishing. Open must come
    first because publish raises on a closed run. Persister-spawn must
    come before publish because — although the broker's `Seq(0)` replay
    guarantees no events are missed — keeping it linear means we never
    have to reason about the alternative.
    """

    broker: EventBroker
    db_sink: DuckDBEventSink
    loop: asyncio.AbstractEventLoop
    _persisters: dict[RunId, asyncio.Task[None]] = field(default_factory=dict)
    _pending_publishes: set[asyncio.Task[Seq]] = field(default_factory=set)

    def emit(self, event: Event) -> None:
        # Single threadsafe hop so the open-or-publish dance happens
        # atomically on the loop thread. Doing the open synchronously
        # in this caller (which may be a non-loop thread, defensively)
        # would race two emits-for-new-run into two open_run calls.
        self.loop.call_soon_threadsafe(self._handle_emit, event)

    def _handle_emit(self, event: Event) -> None:
        rid = RunId(event.run_id)
        if rid not in self._persisters:
            self.broker.open_run(rid)
            self._persisters[rid] = asyncio.create_task(
                persist_to_duckdb_via_sink(self.broker, rid, self.db_sink)
            )
        # Strong reference retained in `_pending_publishes` — without it
        # asyncio may GC the publish task mid-execution. Discard on
        # completion to keep the set bounded. Same pattern as
        # `arena/web/forks.py`'s `fork_tasks`.
        publish_task = asyncio.create_task(self.broker.publish(rid, event))
        self._pending_publishes.add(publish_task)
        publish_task.add_done_callback(self._pending_publishes.discard)

    async def close_all(self) -> None:
        """Drain every queued publish, close every opened run, await every
        persister, then reap.

        Two-tick drain mirrors `arena/web/forks.py::_replay`: the producer
        may have just returned from its last `await`, leaving emit's
        `call_soon_threadsafe` callbacks queued. Tick 1 fires those
        callbacks (which schedule publish tasks); tick 2 lets the publish
        tasks themselves start. Then `gather` on any still-pending
        publishes guarantees they finish before `close_run` — otherwise a
        publish-after-close would raise.

        The trailing `reap` is what makes CLI flows leak-free: a single
        process can run thousands of `synth_game_loop` invocations
        through this sink and never accumulate per-run state past
        `close_all`. The server has a grace-period reaper instead;
        here the process is exiting so no grace is needed.
        """
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if self._pending_publishes:
            await asyncio.gather(*self._pending_publishes)
        for rid in self._persisters:
            self.broker.close_run(rid)
        if self._persisters:
            await asyncio.gather(*self._persisters.values())
        for rid in self._persisters:
            self.broker.reap(rid)


__all__ = [
    "MultiRunBrokerSink",
    "persist_to_duckdb",
    "persist_to_duckdb_via_sink",
]
