"""Integration tests for `evaluation/duckdb_persister.py`.

Covers the persister as a black box: open a broker run, publish events,
verify they all land in DuckDB in `Seq` order via `Event.from_row`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import duckdb

from evaluation.duckdb_persister import (
    MultiRunBrokerSink,
    persist_to_duckdb,
    persist_to_duckdb_via_sink,
)
from evaluation.event_broker import InProcessEventBroker, RunId
from evaluation.event_log import DuckDBEventSink, Event, EventRow, TurnStartPayload

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _read_all(db_path: Path) -> list[Event]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        raw_rows = conn.execute("SELECT * FROM events ORDER BY t").fetchall()
    rows = cast("list[EventRow]", raw_rows)
    return [Event.from_row(row) for row in rows]


def test_persister_writes_every_publishedbuild_event(
    tmp_path: Path, build_event: Callable[..., Event]
) -> None:
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
            await broker.publish(run, build_event("r1", t=i))

        broker.close_run(run)
        await asyncio.wait_for(persist_task, timeout=2.0)

    asyncio.run(scenario())

    persisted = _read_all(db_path)
    assert [e.t for e in persisted] == [0, 1, 2, 3, 4]
    assert all(isinstance(e.payload, TurnStartPayload) for e in persisted)


def test_persister_replays_pre_published_events(
    tmp_path: Path, build_event: Callable[..., Event]
) -> None:
    """Late-spawned persister still drains the full history from seq=0."""
    db_path = tmp_path / "replayed.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        # Publish before the persister exists — broker buffers them.
        for i in range(3):
            await broker.publish(run, build_event("r1", t=i))

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


# ---------------------------------------------------------------------------
# Phase 2.5: MultiRunBrokerSink + persist_to_duckdb_via_sink
# ---------------------------------------------------------------------------


def _read_all_for(db_path: Path, run_id: str) -> list[Event]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        raw_rows = conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY t", [run_id]
        ).fetchall()
    rows = cast("list[EventRow]", raw_rows)
    return [Event.from_row(row) for row in rows]


def test_persist_via_sink_writes_through_shared_conn(
    tmp_path: Path, build_event: Callable[..., Event]
) -> None:
    """`persist_to_duckdb_via_sink` writes via a caller-owned sink — the
    connection stays open across multiple per-run persisters."""
    db_path = tmp_path / "shared.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        with duckdb.connect(str(db_path)) as conn:
            sink = DuckDBEventSink(conn)
            run = RunId("r1")
            broker.open_run(run)

            persist_task = asyncio.create_task(persist_to_duckdb_via_sink(broker, run, sink))
            await asyncio.sleep(0)
            for i in range(3):
                await broker.publish(run, build_event("r1", t=i))
            broker.close_run(run)
            await asyncio.wait_for(persist_task, timeout=2.0)

    asyncio.run(scenario())
    persisted = _read_all_for(db_path, "r1")
    assert [e.t for e in persisted] == [0, 1, 2]


def test_multi_run_sink_auto_opens_runs_on_first_emit(
    tmp_path: Path, build_event: Callable[..., Event]
) -> None:
    """Emitting events for two distinct run_ids must open two broker runs
    and route each event to its run's buffer."""
    db_path = tmp_path / "multi.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        with duckdb.connect(str(db_path)) as conn:
            db_sink = DuckDBEventSink(conn)
            sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop())

            sink.emit(build_event("run_a", t=0))
            sink.emit(build_event("run_b", t=0))
            sink.emit(build_event("run_a", t=1))

            await sink.close_all()
            assert broker.is_open(RunId("run_a")) is False
            assert broker.is_open(RunId("run_b")) is False

    asyncio.run(scenario())
    a_events = _read_all_for(db_path, "run_a")
    b_events = _read_all_for(db_path, "run_b")
    assert [e.t for e in a_events] == [0, 1]
    assert [e.t for e in b_events] == [0]


def test_multi_run_sink_close_all_drains_queued_publishes(
    tmp_path: Path, build_event: Callable[..., Event]
) -> None:
    """Emits queue publishes via `call_soon_threadsafe`; `close_all` must
    wait for them before closing — otherwise the persister exits before
    the last events drain. This is the two-tick-drain invariant."""
    db_path = tmp_path / "drain.duckdb"

    async def scenario() -> None:
        broker = InProcessEventBroker()
        with duckdb.connect(str(db_path)) as conn:
            db_sink = DuckDBEventSink(conn)
            sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop())

            # Emit a burst back-to-back without awaiting between — every
            # publish is queued behind the first call_soon_threadsafe hop.
            for i in range(10):
                sink.emit(build_event("burst", t=i))

            await sink.close_all()

    asyncio.run(scenario())
    persisted = _read_all_for(db_path, "burst")
    # If close_all closed too early, we'd see a truncated list here.
    assert [e.t for e in persisted] == list(range(10))


def test_multi_run_sink_with_race_with_mock_produces_same_events_per_run(
    tmp_path: Path,
) -> None:
    """End-to-end: `race_with_mock` through `MultiRunBrokerSink` results
    in the same per-run event counts as writing through `DuckDBEventSink`
    directly. Proves the broker swap is observationally transparent at
    the CLI call site."""
    from arena.config_profile import ConfigProfile, RaceConfig
    from arena.race import race_with_mock
    from evaluation.world_sim import WorldState

    start_state = WorldState(
        food=200.0,
        wood=150.0,
        gold=0.0,
        stone=0.0,
        population=8,
        pop_cap=25,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )
    config = RaceConfig(
        turns=3,
        profiles=[ConfigProfile(name="mock-a"), ConfigProfile(name="mock-b")],
    )

    direct_db = tmp_path / "direct.duckdb"
    broker_db = tmp_path / "broker.duckdb"

    # Baseline: direct DuckDB sink.
    async def via_direct() -> list[str]:
        with duckdb.connect(str(direct_db)) as conn:
            sink = DuckDBEventSink(conn)
            results = await race_with_mock(config, start_state, sink=sink)
        return [r.loop_result.run_id for r in results]

    direct_run_ids = asyncio.run(via_direct())

    # Phase 2.5: broker + multi-run sink.
    async def via_broker() -> list[str]:
        broker = InProcessEventBroker()
        with duckdb.connect(str(broker_db)) as conn:
            db_sink = DuckDBEventSink(conn)
            sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop())
            try:
                results = await race_with_mock(config, start_state, sink=sink)
            finally:
                await sink.close_all()
        return [r.loop_result.run_id for r in results]

    broker_run_ids = asyncio.run(via_broker())

    # Per-run event counts must match between the two paths. We can't
    # compare run_ids directly (they're freshly minted UUIDs) but we can
    # compare the multiset of counts — same config + deterministic mock
    # invoke means same event totals per run.
    def counts(db_path: Path, run_ids: list[str]) -> list[int]:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            return sorted(
                int(
                    cast(
                        "tuple[int, ...]",
                        conn.execute(
                            "SELECT COUNT(*) FROM events WHERE run_id=?", [rid]
                        ).fetchone(),
                    )[0]
                )
                for rid in run_ids
            )

    assert counts(direct_db, direct_run_ids) == counts(broker_db, broker_run_ids)
