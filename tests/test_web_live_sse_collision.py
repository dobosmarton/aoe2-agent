"""Regression test for the DuckDB connection-mode collision bug.

The original bug (Phase 0): an in-process writer held a child run's
`.duckdb` open read-write while the SSE handler — on the same loop —
opened every file under the logs root read-only to resolve `run_id`.
DuckDB refuses to mix RW and RO connections to the same file in one
process, so the handler crashed with `_duckdb.ConnectionException`.

The Phase 2 fix is structural: live runs never touch DuckDB on the read
side. The handler asks `broker.is_open(run_id)` and streams from memory;
only finalized runs fall through to read-only DuckDB. This test pins
that property at the route boundary:

    1. With the broker reporting the run as open, the `/events` handler
       returns SSE bytes successfully even though a concurrent RW DuckDB
       handle is held on the child file.
    2. The route did not call `duckdb.connect` at all — the structural
       invariant that makes (1) impossible to regress.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import duckdb
from fastapi.responses import StreamingResponse

from arena.web.server import events
from evaluation.event_broker import InProcessEventBroker, RunId
from evaluation.event_log import Event, TurnStartPayload

if TYPE_CHECKING:
    from pathlib import Path


def _event(run_id: str, t: int) -> Event:
    return Event(
        run_id=run_id,
        agent_id="agent-0",
        t=t,
        payload=TurnStartPayload(turn_num=t),
        ts=datetime(2026, 5, 21, 9, 0, 0, tzinfo=UTC),
    )


def test_live_sse_does_not_collide_with_writer(tmp_path: Path) -> None:
    """RW handle held open on the child DB must not break the live SSE
    path. The broker serves the stream from memory; DuckDB is untouched."""
    logs_root = tmp_path / "logs" / "arena"
    day_dir = logs_root / "2026-05-21"
    day_dir.mkdir(parents=True)
    child_db = day_dir / "fork-120000.duckdb"

    run_id = "child-1"
    typed_run = RunId(run_id)

    async def scenario() -> tuple[int, list[str]]:
        broker = InProcessEventBroker()
        broker.open_run(typed_run)
        # Pre-publish events so the stream has something to yield before
        # the close signal arrives.
        for i in range(3):
            await broker.publish(typed_run, _event(run_id, t=i))

        # Hold an RW connection across the SSE iteration — the exact
        # scenario that crashed the old `_live_event_stream` handler.
        # Open inside the async scope so it lives only during streaming.
        holder = duckdb.connect(str(child_db))
        try:
            with patch.object(duckdb, "connect", autospec=True) as connect_spy:
                # Call the route function directly; it returns synchronously
                # because the FastAPI handler is async but the broker branch
                # never awaits before returning the `StreamingResponse`.
                response = await events(run_id=run_id, broker=broker)
                assert isinstance(response, StreamingResponse)
                # `broker.is_open` was True at handler entry, so we're on
                # the broker branch — the route never opened DuckDB.
                lines: list[str] = []

                # Close mid-stream so the broker drains and the iterator
                # terminates cleanly; otherwise it would block waiting for
                # more publishes that never come.
                async def drain() -> None:
                    # StreamingResponse.body_iterator yields str | bytes |
                    # memoryview depending on what the underlying generator
                    # produces; ours yields str, but normalize defensively.
                    async for chunk in response.body_iterator:
                        if isinstance(chunk, str):
                            lines.append(chunk)
                        elif isinstance(chunk, (bytes, memoryview)):
                            lines.append(bytes(chunk).decode())
                        else:
                            raise TypeError(f"unexpected SSE chunk type {type(chunk)!r}")

                drain_task = asyncio.create_task(drain())
                await asyncio.sleep(0)  # let the iterator attach
                broker.close_run(typed_run)
                await asyncio.wait_for(drain_task, timeout=1.0)
                spy_calls = cast("MagicMock", connect_spy).call_count
            return spy_calls, lines
        finally:
            holder.close()

    duckdb_opens, lines = asyncio.run(scenario())
    assert duckdb_opens == 0, "live SSE path opened DuckDB despite an open broker run"
    assert len(lines) == 3
    assert all('"kind":"turn_start"' in line for line in lines)
