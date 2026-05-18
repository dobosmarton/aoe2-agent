"""Unit tests for evaluation/event_log.py (Phase 4).

Uses an in-memory DuckDB connection per test — fast, isolated, no file I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest
from pydantic import TypeAdapter

from evaluation.event_log import (
    SCHEMA_VERSION,
    ActionPayload,
    ActionResultPayload,
    DuckDBEventSink,
    Event,
    ForkPayload,
    LlmPromptPayload,
    LlmResponsePayload,
    MetricPayload,
    NullEventSink,
    ObservationPayload,
    Payload,
    TurnStartPayload,
    WorldMutationPayload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory DuckDB connection. Auto-closed via teardown."""
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def sink(conn: duckdb.DuckDBPyConnection) -> DuckDBEventSink:
    return DuckDBEventSink(conn)


def _event(
    payload: Payload, *, t: int = 0, run_id: str = "run_a", agent_id: str = "agent_x"
) -> Event:
    return Event(
        run_id=run_id,
        agent_id=agent_id,
        t=t,
        payload=payload,
        ts=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
    )


_adapter: TypeAdapter[Payload] = TypeAdapter(Payload)


# ---------------------------------------------------------------------------
# Payload roundtrip via discriminator — Risk 5 mitigation
# ---------------------------------------------------------------------------


def test_turn_start_payload_roundtrips() -> None:
    original = TurnStartPayload(turn_num=5)
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_observation_payload_roundtrips() -> None:
    original = ObservationPayload(entity_count=12, classes=["mill", "town_center"])
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_llm_prompt_payload_roundtrips() -> None:
    original = LlmPromptPayload(state_summary="food=200 wood=150")
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_llm_response_payload_roundtrips() -> None:
    original = LlmResponsePayload(
        actions=[{"type": "queue_villager"}],
        reasoning="economy boost",
        cost_usd=0.0123,
    )
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_action_payload_roundtrips() -> None:
    original = ActionPayload(index_in_turn=0, action={"type": "build", "building_key": "q"})
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_action_result_payload_roundtrips() -> None:
    original = ActionResultPayload(
        index_in_turn=0, action_type="queue_villager", state_changed=True
    )
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_world_mutation_payload_roundtrips() -> None:
    original = WorldMutationPayload(before_summary="pop=5", after_summary="pop=0", reason="chaos")
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_fork_payload_roundtrips() -> None:
    original = ForkPayload(parent_run_id="parent", parent_t=3, mutation_summary="set food=0")
    assert _adapter.validate_json(original.model_dump_json()) == original


def test_metric_payload_roundtrips() -> None:
    original = MetricPayload(name="cost_usd", value=0.045)
    assert _adapter.validate_json(original.model_dump_json()) == original


# ---------------------------------------------------------------------------
# Frozen — payloads cannot be mutated post-construction
# ---------------------------------------------------------------------------


def test_payloads_are_frozen() -> None:
    from pydantic import ValidationError

    payload = TurnStartPayload(turn_num=1)
    with pytest.raises(ValidationError):
        payload.turn_num = 999  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# NullEventSink
# ---------------------------------------------------------------------------


def test_null_event_sink_emit_returns_none() -> None:
    result = NullEventSink().emit(_event(TurnStartPayload(turn_num=0)))
    assert result is None


# ---------------------------------------------------------------------------
# DuckDBEventSink
# ---------------------------------------------------------------------------


def test_duckdb_sink_creates_events_table(conn: duckdb.DuckDBPyConnection) -> None:
    DuckDBEventSink(conn)
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    assert "events" in tables


def test_duckdb_sink_table_creation_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    DuckDBEventSink(conn)
    DuckDBEventSink(conn)  # second construction must not raise
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    assert tables.count("events") == 1


def test_duckdb_sink_emit_inserts_one_row(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=1)))
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1


def test_duckdb_sink_stores_kind(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(MetricPayload(name="cost_usd", value=0.01)))
    kind = conn.execute("SELECT kind FROM events").fetchone()[0]
    assert kind == "metric"


def test_duckdb_sink_stores_run_id(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_xyz"))
    run_id = conn.execute("SELECT run_id FROM events").fetchone()[0]
    assert run_id == "run_xyz"


def test_duckdb_sink_stores_turn(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=4), t=4))
    t = conn.execute("SELECT t FROM events").fetchone()[0]
    assert t == 4


def test_duckdb_sink_stores_schema_version(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0)))
    version = conn.execute("SELECT schema_version FROM events").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_duckdb_sink_payload_json_roundtrips_to_typed_payload(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    original = MetricPayload(name="cost_usd", value=0.04)
    sink.emit(_event(original))
    raw = conn.execute("SELECT payload_json FROM events").fetchone()[0]
    parsed = _adapter.validate_json(raw)
    assert parsed == original


def test_duckdb_sink_appends_multiple_events_in_order(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    for turn in range(5):
        sink.emit(_event(TurnStartPayload(turn_num=turn), t=turn))
    turns = [row[0] for row in conn.execute("SELECT t FROM events ORDER BY t").fetchall()]
    assert turns == [0, 1, 2, 3, 4]


def test_duckdb_sink_query_by_kind_filters_correctly(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0), t=0))
    sink.emit(_event(MetricPayload(name="cost", value=0.01), t=0))
    sink.emit(_event(MetricPayload(name="cost", value=0.02), t=1))
    metric_count = conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'metric'").fetchone()[0]
    assert metric_count == 2


def test_duckdb_sink_separates_runs_by_run_id(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_a"))
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_b"))
    run_a_count = conn.execute("SELECT COUNT(*) FROM events WHERE run_id = 'run_a'").fetchone()[0]
    assert run_a_count == 1
