"""DuckDB persister — one of N event-broker consumers.

This is the structural inversion that makes the architecture work:
nobody but the persister opens the writer's DuckDB file. The SSE handler
reads from the broker for live runs; only the cold path (after
`broker.close_run`) touches DuckDB read-only. The in-process file-coupling
bug disappears as a side effect.

Spawn one persister coroutine per run, alongside the producer:

    broker.open_run(run_id)
    asyncio.create_task(persist_to_duckdb(broker, run_id, db_path))
    # ... producer publishes ...
    broker.close_run(run_id)   # persister drains buffer and returns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from evaluation.event_broker import Seq
from evaluation.event_log import DuckDBEventSink

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.event_broker import EventBroker, RunId


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


__all__ = ["persist_to_duckdb"]
