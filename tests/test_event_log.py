"""Unit tests for evaluation/event_log.py (Phase 4).

Uses an in-memory DuckDB connection per test — fast, isolated, no file I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import duckdb
import pytest
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator

from evaluation.event_log import (
    SCHEMA_VERSION,
    ActionPayload,
    ActionResultPayload,
    DuckDBEventSink,
    Event,
    EventRow,
    ForkPayload,
    LlmPromptPayload,
    LlmResponsePayload,
    MetricPayload,
    NullEventSink,
    ObservationPayload,
    Payload,
    TurnStartPayload,
    WorldMutationPayload,
    WorldStateSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
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


def _scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> object:
    """Execute `sql` and return the first column of the first row.

    Returns `object` (not `Any`) so callers must compare-or-cast — the
    `Any` shortcut would silently green-light bogus operations.
    Assumes the query returns at least one row — asserts otherwise.
    """
    row = conn.execute(sql).fetchone()
    assert row is not None
    # Explicit narrowing — `row[0]` is Any (duckdb returns tuple[Any, ...]);
    # cast to `object` at the boundary so callers can't accidentally rely
    # on Any's silent operation-permissiveness.
    return cast("object", row[0])


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
    assert _scalar(conn, "SELECT COUNT(*) FROM events") == 1


def test_duckdb_sink_stores_kind(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(MetricPayload(name="cost_usd", value=0.01)))
    assert _scalar(conn, "SELECT kind FROM events") == "metric"


def test_duckdb_sink_stores_run_id(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_xyz"))
    assert _scalar(conn, "SELECT run_id FROM events") == "run_xyz"


def test_duckdb_sink_stores_turn(sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=4), t=4))
    assert _scalar(conn, "SELECT t FROM events") == 4


def test_duckdb_sink_stores_schema_version(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0)))
    assert _scalar(conn, "SELECT schema_version FROM events") == SCHEMA_VERSION


def test_duckdb_sink_payload_json_roundtrips_to_typed_payload(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    original = MetricPayload(name="cost_usd", value=0.04)
    sink.emit(_event(original))
    raw = cast("str", _scalar(conn, "SELECT payload_json FROM events"))
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
    assert _scalar(conn, "SELECT COUNT(*) FROM events WHERE kind = 'metric'") == 2


def test_duckdb_sink_separates_runs_by_run_id(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_a"))
    sink.emit(_event(TurnStartPayload(turn_num=0), run_id="run_b"))
    assert _scalar(conn, "SELECT COUNT(*) FROM events WHERE run_id = 'run_a'") == 1


# ---------------------------------------------------------------------------
# Event.from_row — reconstruct an Event from a DuckDB row (inverse of emit)
# ---------------------------------------------------------------------------


# Naive datetime — DuckDB's `ts TIMESTAMP` column drops tzinfo on read-back,
# so equality round-trips cleanly only when the original is also naive.
_TS_NAIVE = datetime(2026, 5, 18, 12, 0, 0)  # noqa: DTZ001 (see comment above)

_ALL_PAYLOAD_KINDS: list[Payload] = [
    TurnStartPayload(turn_num=5),
    ObservationPayload(entity_count=12, classes=["mill", "town_center"]),
    LlmPromptPayload(state_summary="food=200 wood=150"),
    LlmResponsePayload(
        actions=[{"type": "queue_villager"}], reasoning="economy boost", cost_usd=0.0123
    ),
    ActionPayload(index_in_turn=0, action={"type": "build", "building_key": "q"}),
    ActionResultPayload(index_in_turn=0, action_type="queue_villager", state_changed=True),
    WorldMutationPayload(before_summary="pop=5", after_summary="pop=0", reason="chaos"),
    ForkPayload(parent_run_id="parent", parent_t=3, mutation_summary="set food=0"),
    MetricPayload(name="cost_usd", value=0.045),
]


def _emit_and_select(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection, event: Event
) -> EventRow:
    sink.emit(event)
    row = conn.execute(
        "SELECT run_id, agent_id, t, kind, payload_json, ts, schema_version FROM events"
    ).fetchone()
    assert row is not None
    return cast("EventRow", row)


def test_event_from_row_roundtrips_via_duckdb(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    original = Event(
        run_id="r1",
        agent_id="agent_x",
        t=4,
        payload=MetricPayload(name="cost", value=0.01),
        ts=_TS_NAIVE,
    )
    row = _emit_and_select(sink, conn, original)
    assert Event.from_row(row) == original


def _payload_id(payload: Payload) -> str:
    """Typed parametrize-id callback — keeps the lambda out of `Any` territory."""
    return type(payload).__name__


@pytest.mark.parametrize("payload", _ALL_PAYLOAD_KINDS, ids=_payload_id)
def test_event_from_row_roundtrips_all_payload_kinds(
    sink: DuckDBEventSink,
    conn: duckdb.DuckDBPyConnection,
    payload: Payload,
) -> None:
    original = Event(run_id="r1", agent_id="a", t=0, payload=payload, ts=_TS_NAIVE)
    row = _emit_and_select(sink, conn, original)
    assert Event.from_row(row) == original


def test_event_from_row_preserves_non_default_schema_version(
    sink: DuckDBEventSink, conn: duckdb.DuckDBPyConnection
) -> None:
    original = Event(
        run_id="r1",
        agent_id="a",
        t=0,
        payload=TurnStartPayload(turn_num=0),
        ts=_TS_NAIVE,
        schema_version=99,
    )
    row = _emit_and_select(sink, conn, original)
    assert Event.from_row(row).schema_version == 99


# ---------------------------------------------------------------------------
# WorldStateSnapshot (Phase 5)
# ---------------------------------------------------------------------------


def _snapshot(**kwargs) -> WorldStateSnapshot:
    defaults = {
        "food": 200.0,
        "wood": 150.0,
        "gold": 0.0,
        "stone": 0.0,
        "population": 8,
        "pop_cap": 25,
        "age": "Dark Age",
        "buildings": [],
        "villager_queue": [],
        "age_up_ticks_remaining": 0,
        "turn": 0,
    }
    defaults.update(kwargs)
    return WorldStateSnapshot(**defaults)


def _world_state(**kwargs):
    from evaluation.world_sim import WorldState

    defaults = {
        "food": 200.0,
        "wood": 150.0,
        "gold": 0.0,
        "stone": 0.0,
        "population": 8,
        "pop_cap": 25,
        "age": "Dark Age",
        "buildings": [],
        "villager_queue": [],
        "age_up_ticks_remaining": 0,
        "turn": 0,
    }
    defaults.update(kwargs)
    return WorldState(**defaults)


def test_world_state_snapshot_from_world_state_preserves_all_fields() -> None:
    ws = _world_state(food=300.0, wood=200.0, turn=5, buildings=["house"], villager_queue=[2])
    snap = WorldStateSnapshot.from_world_state(ws)
    assert snap.food == 300.0
    assert snap.wood == 200.0
    assert snap.turn == 5
    assert snap.buildings == ["house"]
    assert snap.villager_queue == [2]


def test_world_state_snapshot_to_world_state_inverts_from_world_state() -> None:
    from evaluation.world_sim import WorldState

    ws = _world_state(food=123.5, population=12, age="Feudal Age", buildings=["mill", "house"])
    snap = WorldStateSnapshot.from_world_state(ws)
    restored = snap.to_world_state()
    assert isinstance(restored, WorldState)
    assert restored == ws


def test_world_state_snapshot_roundtrips_via_discriminator() -> None:
    snap = _snapshot(food=400.0, turn=3)
    payload = TurnStartPayload(turn_num=3, state=snap)
    assert _adapter.validate_json(payload.model_dump_json()) == payload


def test_world_state_snapshot_field_parity_with_dataclass() -> None:
    import dataclasses

    from evaluation.world_sim import WorldState

    snapshot_fields = set(WorldStateSnapshot.model_fields.keys())
    dataclass_fields = {f.name for f in dataclasses.fields(WorldState)}
    assert snapshot_fields == dataclass_fields


def test_turn_start_payload_state_defaults_to_none() -> None:
    payload = TurnStartPayload(turn_num=1)
    assert payload.state is None


def test_turn_start_payload_with_state_serializes_and_parses() -> None:
    snap = _snapshot(food=250.0, population=10)
    payload = TurnStartPayload(turn_num=2, state=snap)
    restored = _adapter.validate_json(payload.model_dump_json())
    assert isinstance(restored, TurnStartPayload)
    assert restored.state == snap
