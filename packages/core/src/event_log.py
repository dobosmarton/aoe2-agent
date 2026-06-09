"""Event-log domain types — frozen Pydantic payloads + the EventSink Protocol.

The wire format every consumer reads and every producer writes. DuckDB and
Redis materializations live downstream in `evaluation`; this module is the
contract they implement against.

Schema (versioned via `schema_version` for future migrations):

    events(
        run_id           VARCHAR,
        agent_id         VARCHAR,
        t                INTEGER,           -- turn number, 0-indexed
        kind             VARCHAR,           -- matches Payload.kind discriminator
        payload_json     VARCHAR,           -- Pydantic .model_dump_json()
        ts               TIMESTAMP,         -- wall-clock event time
        schema_version   INTEGER            -- payload schema version
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter

if TYPE_CHECKING:
    from datetime import datetime

    from core.world_state import WorldState


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# WorldStateSnapshot — Pydantic mirror of WorldState for event persistence.
# Embedded in TurnStartPayload so fork() can restore state in O(1).
# ---------------------------------------------------------------------------


class WorldStateSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    food: float
    wood: float
    gold: float
    stone: float
    population: int
    pop_cap: int
    age: str
    buildings: list[str]
    villager_queue: list[int]
    age_up_ticks_remaining: int
    turn: int

    @classmethod
    def from_world_state(cls, state: WorldState) -> WorldStateSnapshot:
        return cls(
            food=state.food,
            wood=state.wood,
            gold=state.gold,
            stone=state.stone,
            population=state.population,
            pop_cap=state.pop_cap,
            age=state.age,
            buildings=list(state.buildings),
            villager_queue=list(state.villager_queue),
            age_up_ticks_remaining=state.age_up_ticks_remaining,
            turn=state.turn,
        )

    def to_world_state(self) -> WorldState:
        # Local import to keep this module free of any cross-module top-level
        # imports beyond stdlib + pydantic. core.world_state has zero deps,
        # so the local import is purely cosmetic — no cycle risk.
        from core.world_state import WorldState as _WorldState

        return _WorldState(
            food=self.food,
            wood=self.wood,
            gold=self.gold,
            stone=self.stone,
            population=self.population,
            pop_cap=self.pop_cap,
            age=self.age,
            buildings=list(self.buildings),
            villager_queue=list(self.villager_queue),
            age_up_ticks_remaining=self.age_up_ticks_remaining,
            turn=self.turn,
        )


# ---------------------------------------------------------------------------
# Payload types — one per `kind`, all frozen.
# ---------------------------------------------------------------------------


class _PayloadBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class TurnStartPayload(_PayloadBase):
    kind: Literal["turn_start"] = "turn_start"
    turn_num: int
    state: WorldStateSnapshot | None = None  # Phase 5: snapshot for fork()
    # Which racing profile/config produced this run. Set on every turn so the
    # last turn_start carries both the final `state` and the config label in a
    # single row. None for runs with no profile (forks, ad-hoc loops).
    profile_name: str | None = None


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
    """Operator-driven mutation. Not emitted by the natural synth loop —
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


# TypeAdapter construction walks the discriminated union once; cache it.
_PAYLOAD_ADAPTER: Final[TypeAdapter[Payload]] = TypeAdapter(Payload)


# Shape of one DuckDB `events` row: matches CREATE_TABLE column order
# (run_id, agent_id, t, kind, payload_json, ts, schema_version). Exported
# so callers reading rows back from the DB can type their helpers
# consistently with Event.from_row.
EventRow = tuple[str, str, int, str, str, "datetime", int]


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

    @classmethod
    def from_row(cls, row: EventRow) -> Event:
        """Reconstruct an Event from a DuckDB `SELECT * FROM events` row.

        Inverse of DuckDBEventSink.emit. The `kind` column is redundant
        with the embedded discriminator in payload_json and is ignored
        here — TypeAdapter validates the discriminator on the payload.
        """
        run_id, agent_id, t, _kind, payload_json, ts, schema_version = row
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            t=t,
            payload=_PAYLOAD_ADAPTER.validate_json(payload_json),
            ts=ts,
            schema_version=schema_version,
        )


# ---------------------------------------------------------------------------
# EventSink Protocol — the producer-side contract.
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    """Where events go. Synchronous; one call per event."""

    def emit(self, event: Event) -> None: ...


class NullEventSink:
    """No-op sink — the default for callers that don't want persistence.
    Singleton-friendly (no state, all method calls return None)."""

    def emit(self, event: Event) -> None:
        return None


__all__ = [
    "SCHEMA_VERSION",
    "_PAYLOAD_ADAPTER",
    "ActionPayload",
    "ActionResultPayload",
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
]
