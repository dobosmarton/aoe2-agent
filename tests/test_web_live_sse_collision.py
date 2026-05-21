"""Regression test for the DuckDB connection-mode collision bug.

Phase 9's async fork replay opens a child run's `.duckdb` file read-write
and holds the connection open for the duration of the replay. When the
frontend immediately opens GET /events?run_id=<child>, the live SSE
handler calls `_resolve_run_optional`, which iterates the logs root and
tries to open every `.duckdb` file with `read_only=True`. DuckDB
disallows mixing access modes for connections to the same file within
one process, and the resolver crashes with:

    _duckdb.ConnectionException: Connection Error: Can't open a
    connection to same database file with a different configuration
    than existing connections

This test reproduces that failure with a precise mock of the writer
(a bare RW connection held open). It is `xfail(strict=True)` until the
event-broker migration (docs/design/event-broker-architecture.md)
lands in Phase 2 — at which point the marker self-graduates: the test
passes, `strict=True` flips it to XPASS-as-failure, and the marker
must be removed.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from evaluation.event_log import DuckDBEventSink, Event, TurnStartPayload

if TYPE_CHECKING:
    from pathlib import Path


def _seed_db(db_path: Path, *, run_id: str) -> None:
    """Create a DuckDB file with one event for `run_id`, then close the conn."""
    conn = duckdb.connect(str(db_path))
    try:
        DuckDBEventSink(conn).emit(
            Event(
                run_id=run_id,
                agent_id="agent-0",
                t=0,
                payload=TurnStartPayload(turn_num=0),
                # Naive datetime — schema column is `ts TIMESTAMP` (naive); see
                # tests/test_event_log.py for the round-trip rationale.
                ts=datetime(2026, 5, 21, 9, 0, 0),  # noqa: DTZ001
            )
        )
    finally:
        conn.close()


@pytest.mark.xfail(
    reason=(
        "DuckDB in-process connection-mode collision; fixed by Phase 2 "
        "event-broker cutover (docs/design/event-broker-architecture.md)."
    ),
    raises=duckdb.Error,
    strict=True,
)
def test_live_sse_does_not_collide_with_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs_root = tmp_path / "logs" / "arena"
    day_dir = logs_root / "2026-05-21"
    day_dir.mkdir(parents=True)
    child_db = day_dir / "fork-120000.duckdb"
    _seed_db(child_db, run_id="child-1")
    monkeypatch.setenv("ARENA_LOGS_ROOT", str(logs_root))

    # Re-import so _logs_root() picks up the env override (project pattern,
    # mirrors tests/test_web_server.py:57-64).
    from arena.web import server as server_module

    importlib.reload(server_module)

    registry = server_module.LiveRunRegistry()
    registry.register("child-1")

    # Hold an RW connection open across the SSE call — precise mock of
    # what the fork replay task does to the child DB file.
    holder = duckdb.connect(str(child_db))
    try:

        async def drive_one_step() -> None:
            gen = server_module._live_event_stream(registry, "child-1")
            async for _ in gen:
                break  # the resolve happens before the first yield

        asyncio.run(drive_one_step())
    finally:
        holder.close()
        registry.finalize("child-1")
