"""Unit guarantee: the broker-source `/events` generator never touches DuckDB.

The Phase 0 regression test in `tests/test_web_live_sse_collision.py`
verifies the property end-to-end through the HTTP layer. This test is
the lower-level unit-scoped equivalent — it proves the structural claim
in isolation: when a run is open in the broker, `_stream_from_broker`
serves it entirely from memory, with zero `duckdb.connect()` calls.
Cheap to maintain, fast to fail loudly if anyone re-introduces a DuckDB
scan in the live SSE path.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import duckdb
from arena_web.server import _stream_from_broker
from evaluation.event_broker import InProcessEventBroker, RunId, Seq

if TYPE_CHECKING:
    from collections.abc import Callable

    from evaluation.event_log import Event


def test_broker_sse_path_avoids_duckdb(build_event: Callable[..., Event]) -> None:
    """`_stream_from_broker` over an open-then-published run must not call
    `duckdb.connect` at all — the entire stream comes from memory.
    This is the structural property the broker cutover stands on."""
    broker = InProcessEventBroker()
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)
        for i in range(3):
            await broker.publish(run, build_event("r1", t=i))
        broker.close_run(run)
        return [line async for line in _stream_from_broker(broker, run)]

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


def test_broker_sse_yields_live_events_during_publish(
    build_event: Callable[..., Event],
) -> None:
    """Live publishes after the consumer attaches must flush to the SSE
    stream without polling. The waiter mechanism is the critical path."""
    broker = InProcessEventBroker()
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)

        async def consumer() -> list[str]:
            return [line async for line in _stream_from_broker(broker, run)]

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let the consumer attach
        for i in range(4):
            await broker.publish(run, build_event("r1", t=i))
        broker.close_run(run)
        return await asyncio.wait_for(task, timeout=1.0)

    lines = asyncio.run(scenario())
    assert len(lines) == 4
    assert all(line.startswith("data: ") for line in lines)


def test_broker_sse_emits_overflow_event_on_backpressure_drop(
    build_event: Callable[..., Event],
) -> None:
    """When the consumer's cursor falls below `head_seq`, the SSE
    generator yields a terminal `event: overflow` line carrying
    `available_from`, then returns — instead of letting the exception
    propagate into Starlette's connection close.

    Frontend semantics: receive `event: overflow`, reconnect with
    `?from_seq=<available_from>`, accept the gap."""
    broker = InProcessEventBroker(max_buffer_size=2)
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, build_event("r1", t=i))
        broker.close_run(run)
        # from_seq=Seq(1) is below head_seq=4 → overflow.
        return [line async for line in _stream_from_broker(broker, run, from_seq=Seq(1))]

    lines = asyncio.run(scenario())
    assert lines, "expected at least the overflow event line"
    overflow_line = lines[-1]
    assert overflow_line.startswith("event: overflow\n")
    assert '"available_from": 4' in overflow_line


def test_broker_sse_honors_from_seq(build_event: Callable[..., Event]) -> None:
    """`from_seq` skips earlier envelopes — basis for the
    `/events?from_seq=K` reconnect-with-cursor recovery path."""
    broker = InProcessEventBroker()
    run = RunId("r1")

    async def scenario() -> list[str]:
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, build_event("r1", t=i))
        broker.close_run(run)
        return [line async for line in _stream_from_broker(broker, run, from_seq=Seq(3))]

    lines = asyncio.run(scenario())
    # Seq(3), Seq(4), Seq(5) — three events.
    assert len(lines) == 3
    assert all(line.startswith("data: ") for line in lines)
