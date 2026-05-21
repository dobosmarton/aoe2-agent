"""Operator-driven fork endpoint + async replay (Phase 9).

`create_fork()` is the entry point called by the FastAPI POST /forks
handler. It snapshot-forks the parent run via evaluation.fork.fork(),
optionally applies a `MutationPatch`, emits a `world_mutation` event,
and schedules a background `synth_game_loop` task to play out N more
turns. The HTTP response returns immediately with the child run_id;
SSE clients subscribe to the live registry to watch events arrive.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from arena.config_profile import ConfigProfile
from arena.invoke import build_synth_invoke
from arena.web.live import BroadcastingSink, LiveRunRegistry
from evaluation.event_log import (
    DuckDBEventSink,
    Event,
    TurnStartPayload,
    WorldMutationPayload,
)
from evaluation.fork import fork
from gameplay_agent.synth_game_loop import synth_game_loop

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.world_sim import WorldState


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
    model="claude-haiku-4-5-20251001",
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
# Background replay task
# ---------------------------------------------------------------------------


async def _replay(
    initial_state: WorldState,
    child_run_id: str,
    n_turns: int,
    api_key: str,
    db_path: Path,
    registry: LiveRunRegistry,
) -> None:
    """Run synth_game_loop, teeing events to DuckDB + the live registry."""
    try:
        with duckdb.connect(str(db_path)) as conn:
            sink = BroadcastingSink(
                db_sink=DuckDBEventSink(conn),
                registry=registry,
                loop=asyncio.get_running_loop(),
            )
            invoke = build_synth_invoke(DEFAULT_FORK_PROFILE, api_key)
            await synth_game_loop(
                invoke=invoke,
                initial_state=initial_state,
                max_iterations=n_turns,
                sink=sink,
                run_id=child_run_id,
            )
    except Exception:
        logger.exception("fork replay failed for run_id=%s", child_run_id)
        raise
    finally:
        # Always close the live channel; otherwise SSE subscribers wait
        # forever on a queue that never receives the None sentinel.
        registry.finalize(child_run_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def create_fork(
    request: ForkRequest,
    api_key: str,
    registry: LiveRunRegistry,
    logs_root: Path,
    fork_tasks: set[asyncio.Task[None]],
) -> ForkResponse:
    """Snapshot the parent at parent_t, optionally mutate, schedule replay.

    Raises FileNotFoundError if the parent run can't be located, or
    ForkError (from evaluation.fork) if parent_t has no turn_start.
    """
    parent_db = _resolve_parent_db(request.parent_run_id, logs_root)
    child_db = _new_child_db_path(logs_root)

    mutation_fn = None if request.mutation.is_empty() else request.mutation.apply

    with duckdb.connect(str(parent_db), read_only=True) as parent_conn:
        parent_state_before = _load_parent_state(
            parent_conn, request.parent_run_id, request.parent_t
        )
        with duckdb.connect(str(child_db)) as child_conn:
            db_sink = DuckDBEventSink(child_conn)
            child_run_id, forked_state = fork(
                conn=parent_conn,
                parent_run_id=request.parent_run_id,
                parent_t=request.parent_t,
                sink=db_sink,
                mutation_fn=mutation_fn,
            )
            if mutation_fn is not None and parent_state_before is not None:
                db_sink.emit(
                    _build_world_mutation_event(
                        child_run_id=child_run_id,
                        parent_t=request.parent_t,
                        before=parent_state_before,
                        after=forked_state,
                        reason=request.reason,
                    )
                )

    registry.register(child_run_id)
    task = asyncio.create_task(
        _replay(
            initial_state=forked_state,
            child_run_id=child_run_id,
            n_turns=request.n_turns,
            api_key=api_key,
            db_path=child_db,
            registry=registry,
        )
    )
    # Retain a strong reference to the task — without this, asyncio's
    # garbage collector can drop the task mid-execution.
    fork_tasks.add(task)
    task.add_done_callback(fork_tasks.discard)

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
