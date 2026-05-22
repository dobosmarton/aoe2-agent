"""Phase 1 integration test: the broker-on `/events` path never touches DuckDB.

This is *not* the Phase 0 collision-regression test. That one stays xfail
until Phase 2 (it exercises `_live_event_stream` directly, which the broker
flag doesn't route past). What this test proves is the structural claim
of Phase 1: when a run is open in the broker, the new `_broker_sse` path
serves it entirely from the in-memory broker — zero `duckdb.connect()`
calls. That property is the foundation the Phase 2 cutover stands on.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import duckdb

from arena.web.server import _broker_sse
from evaluation.event_broker import InProcessEventBroker, RunId
from evaluation.event_log import Event, TurnStartPayload


def _event(run_id: str, t: int) -> Event:
    return Event(
        run_id=run_id,
        agent_id="agent_x",
        t=t,
        payload=TurnStartPayload(turn_num=t),
        ts=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
    )


def test_broker_sse_path_avoids_duckdb() -> None:
    """`_broker_sse` over an open-then-published run must not call
    `duckdb.connect` at all — the entire stream comes from memory.
    This is the structural property that makes Phase 2's cutover safe."""
    broker = InProcessEventBroker()
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)
        for i in range(3):
            await broker.publish(run, _event("r1", t=i))
        broker.close_run(run)
        return [line async for line in _broker_sse(broker, "r1")]

    with patch.object(duckdb, "connect", autospec=True) as connect_spy:
        lines = asyncio.run(scenario())

    assert connect_spy.call_count == 0, (
        "broker-on path opened DuckDB despite the run being in the broker"
    )
    assert len(lines) == 3
    assert all(line.startswith("data: ") for line in lines)
    assert all(line.endswith("\n\n") for line in lines)
    # The wire payload contains the discriminator + turn_num the frontend reads.
    assert all('"kind":"turn_start"' in line for line in lines)


def test_broker_sse_yields_live_events_during_publish() -> None:
    """Live publishes after the consumer attaches must flush to the SSE
    stream without polling. The waiter mechanism is the critical path."""
    broker = InProcessEventBroker()
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)

        async def consumer() -> list[str]:
            return [line async for line in _broker_sse(broker, "r1")]

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let the consumer attach
        for i in range(4):
            await broker.publish(run, _event("r1", t=i))
        broker.close_run(run)
        return await asyncio.wait_for(task, timeout=1.0)

    lines = asyncio.run(scenario())
    assert len(lines) == 4
    assert all(line.startswith("data: ") for line in lines)
