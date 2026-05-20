"""Unit tests for evaluation/fork.py (Phase 5).

Uses in-memory DuckDB plus the real synth_game_loop to build parent runs.
All tests are offline and deterministic.
"""

from __future__ import annotations

import asyncio

import duckdb
import pytest

from evaluation.event_log import DuckDBEventSink, Event, NullEventSink, TurnStartPayload
from evaluation.fork import ForkError, fork
from evaluation.world_sim import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


class _RecordingSink:
    def __init__(self):
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _initial_state() -> WorldState:
    return WorldState(
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


class _EmptyStub:
    async def __call__(self, state: WorldState) -> tuple[list[dict], str, float]:
        return ([], "", 0.0)


def _run_parent(conn: duckdb.DuckDBPyConnection, turns: int = 3) -> str:
    """Run synth_game_loop into DuckDB and return run_id."""
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = DuckDBEventSink(conn)
    result = asyncio.run(
        synth_game_loop(_EmptyStub(), _initial_state(), max_iterations=turns, sink=sink)
    )
    return result.run_id


# ---------------------------------------------------------------------------
# fork() — identity (no mutation)
# ---------------------------------------------------------------------------


def test_fork_returns_new_run_id_hex_length_32(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    new_run_id, _ = fork(conn, run_id, parent_t=1, sink=NullEventSink())
    assert len(new_run_id) == 32


def test_fork_new_run_id_differs_from_parent(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    new_run_id, _ = fork(conn, run_id, parent_t=1, sink=NullEventSink())
    assert new_run_id != run_id


def test_fork_emits_one_fork_event(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    sink = _RecordingSink()
    fork(conn, run_id, parent_t=1, sink=sink)
    assert len(sink.events) == 1


def test_fork_payload_references_parent_run_id(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    sink = _RecordingSink()
    fork(conn, run_id, parent_t=1, sink=sink)
    assert sink.events[0].payload.parent_run_id == run_id  # type: ignore[union-attr]


def test_fork_payload_references_parent_t(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    sink = _RecordingSink()
    fork(conn, run_id, parent_t=2, sink=sink)
    assert sink.events[0].payload.parent_t == 2  # type: ignore[union-attr]


def test_fork_state_matches_parent_turn_start_when_no_mutation(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    run_id = _run_parent(conn, turns=3)
    _, forked_state = fork(conn, run_id, parent_t=1, sink=NullEventSink())
    # t=1 stores the initial state (food=200, before any ticks)
    assert forked_state.food == 200.0


def test_fork_without_mutation_fn_leaves_mutation_summary_empty(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    run_id = _run_parent(conn)
    sink = _RecordingSink()
    fork(conn, run_id, parent_t=1, sink=sink)
    assert sink.events[0].payload.mutation_summary == ""  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# fork() — with mutation
# ---------------------------------------------------------------------------


def test_fork_with_mutation_fn_applies_to_state(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    from dataclasses import replace

    mutate = lambda s: replace(s, food=0.0)  # noqa: E731
    _, forked_state = fork(conn, run_id, parent_t=1, sink=NullEventSink(), mutation_fn=mutate)
    assert forked_state.food == 0.0


def test_fork_with_mutation_fn_records_diff_summary(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn)
    from dataclasses import replace

    mutate = lambda s: replace(s, food=0.0)  # noqa: E731
    sink = _RecordingSink()
    fork(conn, run_id, parent_t=1, sink=sink, mutation_fn=mutate)
    summary = sink.events[0].payload.mutation_summary  # type: ignore[union-attr]
    assert "food" in summary


# ---------------------------------------------------------------------------
# fork() — error cases
# ---------------------------------------------------------------------------


def test_fork_raises_fork_error_when_no_turn_start_at_t(conn: duckdb.DuckDBPyConnection) -> None:
    run_id = _run_parent(conn, turns=3)
    with pytest.raises(ForkError):
        fork(conn, run_id, parent_t=99, sink=NullEventSink())


def test_fork_raises_fork_error_when_turn_start_lacks_state(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # Insert a Phase-4-style turn_start with state=None (legacy event without snapshot).
    from datetime import UTC, datetime

    from evaluation.event_log import Event

    legacy_sink = DuckDBEventSink(conn)
    legacy_run_id = "legacy-run-001"
    legacy_sink.emit(
        Event(
            run_id=legacy_run_id,
            agent_id="agent_x",
            t=1,
            payload=TurnStartPayload(turn_num=1),  # no state field → state=None
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    with pytest.raises(ForkError):
        fork(conn, legacy_run_id, parent_t=1, sink=NullEventSink())


# ---------------------------------------------------------------------------
# fork() — end-to-end continuation
# ---------------------------------------------------------------------------


def test_fork_end_to_end_continuation(conn: duckdb.DuckDBPyConnection) -> None:
    from gameplay_agent.synth_game_loop import synth_game_loop

    # Parent: 3 empty turns.
    parent_run_id = _run_parent(conn, turns=3)

    # Fork at t=2 (state after turn 1 tick: food=220, turn=1).
    fork_sink = DuckDBEventSink(conn)
    new_run_id, forked_state = fork(conn, parent_run_id, parent_t=2, sink=fork_sink)

    # Continue for 2 more empty turns from the forked state.
    child_sink = DuckDBEventSink(conn)
    child_result = asyncio.run(
        synth_game_loop(
            _EmptyStub(),
            forked_state,
            max_iterations=2,
            sink=child_sink,
            run_id=new_run_id,
        )
    )

    # 2 ticks from forked_state.turn advance the turn counter by 2.
    assert child_result.final_state.turn == forked_state.turn + 2
