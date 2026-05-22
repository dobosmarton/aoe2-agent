"""Integration tests for `evaluation/duckdb_persister.py`.

Covers the persister as a black box: open a broker run, publish events,
verify they all land in DuckDB in `Seq` order via `Event.from_row`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import duckdb

from evaluation.duckdb_persister import persist_to_duckdb
from evaluation.event_broker import InProcessEventBroker, RunId
from evaluation.event_log import Event, EventRow, TurnStartPayload

if TYPE_CHECKING:
    from pathlib import Path


def _event(run_id: str, t: int) -> Event:
    return Event(
        run_id=run_id,
        agent_id="agent_x",
        t=t,
        payload=TurnStartPayload(turn_num=t),
        ts=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
    )


def _read_all(db_path: Path) -> list[Event]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        raw_rows = conn.execute("SELECT * FROM events ORDER BY t").fetchall()
    rows = cast("list[EventRow]", raw_rows)
    return [Event.from_row(row) for row in rows]


def test_persister_writes_every_published_event(tmp_path: Path) -> None:
    """Producer + persister run concurrently; persister mirrors every event."""
    db_path = tmp_path / "persisted.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        persist_task = asyncio.create_task(persist_to_duckdb(broker, run, db_path))
        # Let the persister attach its broker subscription first.
        await asyncio.sleep(0)

        for i in range(5):
            await broker.publish(run, _event("r1", t=i))

        broker.close_run(run)
        await asyncio.wait_for(persist_task, timeout=2.0)

    asyncio.run(scenario())

    persisted = _read_all(db_path)
    assert [e.t for e in persisted] == [0, 1, 2, 3, 4]
    assert all(isinstance(e.payload, TurnStartPayload) for e in persisted)


def test_persister_replays_pre_published_events(tmp_path: Path) -> None:
    """Late-spawned persister still drains the full history from seq=0."""
    db_path = tmp_path / "replayed.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        # Publish before the persister exists — broker buffers them.
        for i in range(3):
            await broker.publish(run, _event("r1", t=i))

        persist_task = asyncio.create_task(persist_to_duckdb(broker, run, db_path))
        broker.close_run(run)
        await asyncio.wait_for(persist_task, timeout=2.0)

    asyncio.run(scenario())
    persisted = _read_all(db_path)
    assert [e.t for e in persisted] == [0, 1, 2]


def test_persister_exits_when_run_closes(tmp_path: Path) -> None:
    """Persister's `async for` must terminate cleanly after close_run."""
    db_path = tmp_path / "empty.duckdb"

    async def scenario() -> float:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        persist_task = asyncio.create_task(persist_to_duckdb(broker, run, db_path))
        await asyncio.sleep(0)  # let persister subscribe
        broker.close_run(run)

        start = asyncio.get_running_loop().time()
        await asyncio.wait_for(persist_task, timeout=1.0)
        return asyncio.get_running_loop().time() - start

    elapsed = asyncio.run(scenario())
    # Persister should return almost immediately on close; 100ms is generous.
    assert elapsed < 0.1
