"""Event log for the synthetic-arena (Phase 4).

DuckDB-backed event store with versioned Pydantic payloads. The store is a
single file (or in-memory), trivial to ship as a fixture or replay artifact.
Mirroring to Langfuse (per design doc §C) is deferred to Phase 10+.

Schema (versioned via `schema_version` column for future migrations):

    events(
        run_id           VARCHAR,
        agent_id         VARCHAR,
        t                INTEGER,           -- turn number, 0-indexed
        kind             VARCHAR,           -- matches Payload.kind discriminator
        payload_json     VARCHAR,           -- Pydantic .model_dump_json()
        ts               TIMESTAMP,         -- wall-clock event time
        schema_version   INTEGER            -- payload schema version (currently 1)
    )

Covers the 9 event kinds in design doc §C:
    turn_start, observation, llm_prompt, llm_response,
    action, action_result, world_mutation, fork, metric

Phase 4's `synth_game_loop` emits 6 of those 9 — observation, world_mutation,
and fork need data the synth loop does not own today (observation needs
entity rendering, world_mutation needs an operator-mutate hook per design
doc §G, fork is Phase 5). Their payload shapes are defined here so the
schema is lock-in-stable from day one (Risk 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Discriminator

if TYPE_CHECKING:
    from datetime import datetime

    import duckdb


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Payload types — one per `kind` in §C, all frozen.
# ---------------------------------------------------------------------------


class _PayloadBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class TurnStartPayload(_PayloadBase):
    kind: Literal["turn_start"] = "turn_start"
    turn_num: int


class ObservationPayload(_PayloadBase):
    kind: Literal["observation"] = "observation"
    entity_count: int
    classes: list[str]  # sorted unique class_names from the observation


class LlmPromptPayload(_PayloadBase):
    kind: Literal["llm_prompt"] = "llm_prompt"
    state_summary: str  # human-readable WorldState snapshot the LLM sees


class LlmResponsePayload(_PayloadBase):
    kind: Literal["llm_response"] = "llm_response"
    actions: list[dict[str, object]]
    reasoning: str
    cost_usd: float


class ActionPayload(_PayloadBase):
    kind: Literal["action"] = "action"
    index_in_turn: int
    action: dict[str, object]


class ActionResultPayload(_PayloadBase):
    kind: Literal["action_result"] = "action_result"
    index_in_turn: int
    action_type: str
    state_changed: bool


class WorldMutationPayload(_PayloadBase):
    """Operator-driven mutation per design doc §G (chaos mode) and §A
    (mutate() perturbations). Not emitted by the natural synth loop —
    only by explicit operator/chaos hooks. Phase 9."""

    kind: Literal["world_mutation"] = "world_mutation"
    before_summary: str
    after_summary: str
    reason: str = ""


class ForkPayload(_PayloadBase):
    """Spawned a child run from this point. Phase 5."""

    kind: Literal["fork"] = "fork"
    parent_run_id: str
    parent_t: int
    mutation_summary: str = ""


class MetricPayload(_PayloadBase):
    kind: Literal["metric"] = "metric"
    name: str
    value: float


Payload = Annotated[
    TurnStartPayload
    | ObservationPayload
    | LlmPromptPayload
    | LlmResponsePayload
    | ActionPayload
    | ActionResultPayload
    | WorldMutationPayload
    | ForkPayload
    | MetricPayload,
    Discriminator("kind"),
]


# ---------------------------------------------------------------------------
# Event wrapper — what gets persisted.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    run_id: str
    agent_id: str
    t: int
    payload: Payload
    ts: datetime
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Sink protocol + implementations.
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    """Where events go. Synchronous; one call per event."""

    def emit(self, event: Event) -> None: ...


class NullEventSink:
    """No-op sink — the default for callers that don't want persistence.
    Singleton-friendly (no state, all method calls return None)."""

    def emit(self, event: Event) -> None:
        return None


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
