"""Operator-driven fork endpoint + async replay (Phase 9, broker-wired Phase 2).

`create_fork()` is the entry point called by the FastAPI POST /forks
handler. It snapshot-forks the parent run via evaluation.fork.fork(),
optionally applies a `MutationPatch`, publishes a `world_mutation` event,
and schedules a background `synth_game_loop` task to play out N more
turns. Every event flows through the EventBroker — SSE clients consume
the broker; the persister mirrors to DuckDB.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import duckdb
from arena.config_profile import ConfigProfile
from arena.invoke import build_synth_invoke
from evaluation.duckdb_persister import persist_to_duckdb
from evaluation.event_broker import BrokerEventSink, EventBroker, RunId
from evaluation.event_log import (
    Event,
    TurnStartPayload,
    WorldMutationPayload,
)
from evaluation.fork import fork
from gameplay_agent.synth_game_loop import synth_game_loop
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from evaluation.world_sim import WorldState


def _null_on_close(_: RunId) -> None:
    """Default `on_close` callback — no-op for CLI and test paths."""


logger = logging.getLogger(__name__)


_AgeLiteral = Literal["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]
_TURN_START_ADAPTER: TypeAdapter[TurnStartPayload] = TypeAdapter(TurnStartPayload)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class MutationPatch(BaseModel):
    """Optional per-field overrides applied to the forked WorldState."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    food: float | None = None
    wood: float | None = None
    gold: float | None = None
    stone: float | None = None
    population: int | None = Field(default=None, ge=0)
    pop_cap: int | None = Field(default=None, ge=0)
    age: _AgeLiteral | None = None

    def is_empty(self) -> bool:
        return self.model_dump(exclude_none=True) == {}

    def apply(self, state: WorldState) -> WorldState:
        return replace(state, **self.model_dump(exclude_none=True))


class ForkRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_run_id: str = Field(min_length=1)
    parent_t: int = Field(ge=0)
    mutation: MutationPatch = MutationPatch()
    n_turns: int = Field(default=10, ge=0, le=200)
    reason: str = ""


class ForkResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    child_run_id: str
    db_path: str
    profile_used: str


DEFAULT_FORK_PROFILE = ConfigProfile(
    name="operator-fork",
    model="gpt-5.6-luna",
    temperature=0.5,
    prompt_variant="strategy",
)


def _profile_label(profile: ConfigProfile) -> str:
    return f"{profile.model} / {profile.prompt_variant} / temp={profile.temperature}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_child_db_path(logs_root: Path) -> Path:
    now = datetime.now(UTC)
    day_dir = logs_root / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"fork-{now.strftime('%H%M%S%f')}.duckdb"


def _resolve_parent_db(parent_run_id: str, logs_root: Path) -> Path:
    """Locate the .duckdb file containing parent_run_id; newest-first scan."""
    if not logs_root.exists():
        raise FileNotFoundError(f"logs root {logs_root} does not exist")
    candidates = sorted(
        logs_root.glob("*/*.duckdb"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for db_path in candidates:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE run_id=? LIMIT 1", [parent_run_id]
            ).fetchone()
        if row is not None:
            return db_path
    raise FileNotFoundError(f"parent run_id {parent_run_id!r} not found in {logs_root}")


def _load_parent_state(
    conn: duckdb.DuckDBPyConnection, parent_run_id: str, parent_t: int
) -> WorldState | None:
    row = conn.execute(
        "SELECT payload_json FROM events WHERE run_id=? AND t=? AND kind='turn_start' LIMIT 1",
        [parent_run_id, parent_t],
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        return None
    payload = _TURN_START_ADAPTER.validate_json(row[0])
    return payload.state.to_world_state() if payload.state is not None else None


def _state_summary(state: WorldState) -> str:
    return (
        f"food={int(state.food)} wood={int(state.wood)} "
        f"pop={state.population}/{state.pop_cap} age={state.age}"
    )


def _build_world_mutation_event(
    child_run_id: str,
    parent_t: int,
    before: WorldState,
    after: WorldState,
    reason: str,
) -> Event:
    return Event(
        run_id=child_run_id,
        agent_id="",
        t=parent_t,
        payload=WorldMutationPayload(
            before_summary=_state_summary(before),
            after_summary=_state_summary(after),
            reason=reason,
        ),
        ts=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Producer helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CapturingSink:
    """In-memory EventSink that just appends emits to a list.

    Used to bridge `evaluation.fork.fork()` (sync `EventSink` API) into
    the async broker — capture the snapshot event(s), then `await
    broker.publish(...)` for each in the async caller.
    """

    events: list[Event] = field(default_factory=list)

    def emit(self, event: Event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Background replay task
# ---------------------------------------------------------------------------


async def _replay(
    initial_state: WorldState,
    child_run_id: str,
    n_turns: int,
    broker: EventBroker,
    persist_task: asyncio.Task[None],
    on_close: Callable[[RunId], None],
) -> None:
    """Run synth_game_loop publishing through the broker; close on exit.

    Lifecycle ordering at the tail is load-bearing:
        1. drain pending `BrokerEventSink.emit` publishes (they go via
           `call_soon_threadsafe` so are still queued when the game loop
           returns from its last `await`).
        2. `broker.close_run` — signals the persister to drain and exit.
        3. `await persist_task` — guarantees DuckDB is fully written
           before any cold-path reader sees the run as finalized.
        4. `on_close(typed_run)` — notify the server's reaper registry
           so the buffer can be reaped after the grace period.
    DO NOT REORDER — see the broker-architecture design doc § "Subtle
    correctness items".
    """
    typed_run = RunId(child_run_id)
    try:
        sink = BrokerEventSink(
            broker=broker,
            run_id=typed_run,
            loop=asyncio.get_running_loop(),
        )
        invoke = build_synth_invoke(DEFAULT_FORK_PROFILE)
        await synth_game_loop(
            invoke=invoke,
            initial_state=initial_state,
            max_iterations=n_turns,
            sink=sink,
            run_id=child_run_id,
        )
        # Two-tick drain: BrokerEventSink does
        # `call_soon_threadsafe(create_task, broker.publish(...))`.
        # Tick 1 fires the queued callbacks (which schedule the publish
        # tasks). Tick 2 lets the publish tasks themselves run (their
        # bodies are sync, so they complete inside one tick).
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    except Exception:
        logger.exception("fork replay failed for run_id=%s", child_run_id)
        raise
    finally:
        broker.close_run(typed_run)
        try:
            await persist_task
        except Exception:
            logger.exception("persister failed for run_id=%s", child_run_id)
        on_close(typed_run)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def create_fork(
    request: ForkRequest,
    broker: EventBroker,
    logs_root: Path,
    fork_tasks: set[asyncio.Task[None]],
    on_close: Callable[[RunId], None] = _null_on_close,
) -> ForkResponse:
    """Snapshot the parent at parent_t, optionally mutate, schedule replay.

    Lifecycle ordering at the head is load-bearing:
        1. `broker.open_run` — must come before any publish or
           persister subscription.
        2. `await broker.publish(snapshot+mutation events)` — synchronous
           publishes inside the async handler, so they land in the
           buffer in deterministic order.
        3. spawn `persist_to_duckdb` task — subscribes from `Seq(0)`,
           so order doesn't matter for it specifically, but spawning
           after the head publishes keeps the call sequence linear.
        4. spawn `_replay` task — the replay's emits race against any
           late subscriber, which is fine: the broker buffer covers it.

    Raises FileNotFoundError if the parent run can't be located, or
    ForkError (from evaluation.fork) if parent_t has no turn_start.
    """
    parent_db = _resolve_parent_db(request.parent_run_id, logs_root)
    child_db = _new_child_db_path(logs_root)

    mutation_fn = None if request.mutation.is_empty() else request.mutation.apply
    captured = _CapturingSink()

    with duckdb.connect(str(parent_db), read_only=True) as parent_conn:
        parent_state_before = _load_parent_state(
            parent_conn, request.parent_run_id, request.parent_t
        )
        child_run_id, forked_state = fork(
            conn=parent_conn,
            parent_run_id=request.parent_run_id,
            parent_t=request.parent_t,
            sink=captured,
            mutation_fn=mutation_fn,
        )

    typed_run = RunId(child_run_id)
    broker.open_run(typed_run)
    for event in captured.events:
        await broker.publish(typed_run, event)
    if mutation_fn is not None and parent_state_before is not None:
        await broker.publish(
            typed_run,
            _build_world_mutation_event(
                child_run_id=child_run_id,
                parent_t=request.parent_t,
                before=parent_state_before,
                after=forked_state,
                reason=request.reason,
            ),
        )

    persist_task = asyncio.create_task(persist_to_duckdb(broker, typed_run, child_db))
    fork_tasks.add(persist_task)
    persist_task.add_done_callback(fork_tasks.discard)

    replay_task = asyncio.create_task(
        _replay(
            initial_state=forked_state,
            child_run_id=child_run_id,
            n_turns=request.n_turns,
            broker=broker,
            persist_task=persist_task,
            on_close=on_close,
        )
    )
    fork_tasks.add(replay_task)
    replay_task.add_done_callback(fork_tasks.discard)

    return ForkResponse(
        child_run_id=child_run_id,
        db_path=str(child_db),
        profile_used=_profile_label(DEFAULT_FORK_PROFILE),
    )


__all__ = [
    "DEFAULT_FORK_PROFILE",
    "ForkRequest",
    "ForkResponse",
    "MutationPatch",
    "create_fork",
]
