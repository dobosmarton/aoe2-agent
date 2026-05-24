"""DuckDB sink + cold-path reader for the event log.

The event-log *types* (Event, Payload, EventSink Protocol, etc.) live in
`core.event_log`. This module is the DuckDB-bound consumer: it owns the
table schema, the INSERT statement, and the post-finalize `stream_cold`
reader. Splitting the types out of here removed the back-edge that forced
every consumer to depend on duckdb just to use Event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# Re-exports kept for backwards compatibility within `evaluation`. New
# consumers should `from core.event_log import ...` directly.
from core.event_log import (
    SCHEMA_VERSION,
    ActionPayload,
    ActionResultPayload,
    Event,
    EventRow,
    EventSink,
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

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import duckdb
    from evaluation.event_broker import EventEnvelope


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    run_id           VARCHAR,
    agent_id         VARCHAR,
    t                INTEGER,
    kind             VARCHAR,
    payload_json     VARCHAR,
    ts               TIMESTAMP,
    schema_version   INTEGER
)
"""

_INSERT_SQL = """
INSERT INTO events (run_id, agent_id, t, kind, payload_json, ts, schema_version)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class DuckDBEventSink:
    """Append events to a DuckDB connection.

    The events table is created on first use (idempotent). The caller owns
    the connection's lifecycle — closes it, attaches multiple sinks to the
    same connection, etc.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._conn.execute(_CREATE_TABLE_SQL)

    def emit(self, event: Event) -> None:
        self._conn.execute(
            _INSERT_SQL,
            (
                event.run_id,
                event.agent_id,
                event.t,
                event.payload.kind,
                event.payload.model_dump_json(),
                event.ts,
                event.schema_version,
            ),
        )


# ---------------------------------------------------------------------------
# Cold-path reader (post-finalize) — mirrors the broker's EventEnvelope contract.
# ---------------------------------------------------------------------------


def stream_cold(db_path: Path, run_id: str) -> Iterator[EventEnvelope]:
    """Read finalized events for `run_id` from DuckDB, in canonical order.

    Assigns `Seq` from row order (1-indexed) so cold readers see the same
    envelope shape the broker delivers live. `ORDER BY t, rowid` is
    load-bearing: same-turn events share `t`, and `rowid` is DuckDB's
    stable insert-order tiebreak — without it, replays would silently
    reorder events that share a turn number.

    Caller must guarantee no in-process writer holds `db_path` RW; this
    function opens read-only.
    """
    # Local import: `evaluation.event_broker` depends on `Event` from this
    # module (re-exported from core), so importing it at module scope
    # creates a cycle.
    import duckdb
    from evaluation.event_broker import EventEnvelope, RunId, Seq

    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = cast(
            "list[EventRow]",
            conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY t, rowid",
                [run_id],
            ).fetchall(),
        )
    typed_run = RunId(run_id)
    for i, row in enumerate(rows, start=1):
        yield EventEnvelope(run_id=typed_run, seq=Seq(i), event=Event.from_row(row))


__all__ = [
    "SCHEMA_VERSION",
    "ActionPayload",
    "ActionResultPayload",
    "DuckDBEventSink",
    "Event",
    "EventRow",
    "EventSink",
    "ForkPayload",
    "LlmPromptPayload",
    "LlmResponsePayload",
    "MetricPayload",
    "NullEventSink",
    "ObservationPayload",
    "Payload",
    "TurnStartPayload",
    "WorldMutationPayload",
    "WorldStateSnapshot",
    "stream_cold",
]
