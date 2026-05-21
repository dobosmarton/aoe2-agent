"""Tests for arena/web/live.py + arena/web/forks.py (Phase 9). Offline."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest
from pydantic import ValidationError

from arena.web import forks as forks_module
from arena.web.forks import ForkRequest, MutationPatch, create_fork
from arena.web.live import LiveRunRegistry
from evaluation.event_log import (
    DuckDBEventSink,
    Event,
    MetricPayload,
    TurnStartPayload,
    WorldStateSnapshot,
)
from evaluation.world_sim import WorldState

if TYPE_CHECKING:
    from pathlib import Path


def _state(food: float = 200.0, age: str = "Dark Age", pop: int = 8) -> WorldState:
    return WorldState(
        food=food,
        wood=150.0,
        gold=0.0,
        stone=0.0,
        population=pop,
        pop_cap=25,
        age=age,
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def _make_parent_log(db_path: Path, run_id: str, state: WorldState) -> None:
    """Fabricate a parent DuckDB log with one turn_start at t=1."""
    with duckdb.connect(str(db_path)) as conn:
        sink = DuckDBEventSink(conn)
        sink.emit(
            Event(
                run_id=run_id,
                agent_id="agent-0",
                t=1,
                payload=TurnStartPayload(
                    turn_num=1,
                    state=WorldStateSnapshot.from_world_state(state),
                ),
                ts=datetime.now(UTC),
            )
        )


@pytest.fixture
def logs_root(tmp_path: Path) -> Path:
    root = tmp_path / "logs" / "arena"
    (root / "2026-05-21").mkdir(parents=True)
    return root


def _patch_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap fork's real-Claude invoke for the offline mock."""
    from arena.invoke import build_mock_invoke

    monkeypatch.setattr(
        forks_module,
        "build_synth_invoke",
        lambda _profile, _api_key: build_mock_invoke(),
    )


# ---------------------------------------------------------------------------
# MutationPatch
# ---------------------------------------------------------------------------


def test_mutation_patch_is_empty_when_all_none() -> None:
    assert MutationPatch().is_empty() is True


def test_mutation_patch_applies_food() -> None:
    patched = MutationPatch(food=999.0).apply(_state())
    assert patched.food == 999.0


def test_mutation_patch_applies_age() -> None:
    patched = MutationPatch(age="Feudal Age").apply(_state())
    assert patched.age == "Feudal Age"


def test_mutation_patch_rejects_invalid_age() -> None:
    with pytest.raises(ValidationError):
        MutationPatch(age="Imperial Age 2")  # pyright: ignore[reportArgumentType]


def test_mutation_patch_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MutationPatch(food=10.0, mystery_field=5)  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# LiveRunRegistry
# ---------------------------------------------------------------------------


def test_registry_subscriber_receives_published_event() -> None:
    async def run() -> Event | None:
        registry = LiveRunRegistry()
        registry.register("R1")
        sub = registry.subscribe("R1")
        event = Event(
            run_id="R1",
            agent_id="a",
            t=1,
            payload=MetricPayload(name="x", value=1.0),
            ts=datetime.now(UTC),
        )
        registry.publish_nowait(event)
        return await asyncio.wait_for(sub.queue.get(), timeout=1.0)

    received = asyncio.run(run())
    assert received is not None and received.run_id == "R1"


def test_registry_finalize_sends_none_sentinel() -> None:
    async def run() -> Event | None:
        registry = LiveRunRegistry()
        registry.register("R2")
        sub = registry.subscribe("R2")
        registry.finalize("R2")
        return await asyncio.wait_for(sub.queue.get(), timeout=1.0)

    assert asyncio.run(run()) is None


def test_registry_is_live_returns_false_after_finalize() -> None:
    registry = LiveRunRegistry()
    registry.register("R3")
    registry.finalize("R3")
    assert registry.is_live("R3") is False


# ---------------------------------------------------------------------------
# create_fork
# ---------------------------------------------------------------------------


def test_create_fork_raises_when_parent_run_missing(logs_root: Path) -> None:
    request = ForkRequest(parent_run_id="ghost", parent_t=1)
    with pytest.raises(FileNotFoundError):
        asyncio.run(create_fork(request, "stub-key", LiveRunRegistry(), logs_root, set()))


def test_create_fork_returns_child_run_id(
    logs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_invoke(monkeypatch)
    _make_parent_log(logs_root / "2026-05-21" / "race-100000.duckdb", "P1", _state())

    async def go() -> str:
        registry = LiveRunRegistry()
        tasks: set[asyncio.Task[None]] = set()
        response = await create_fork(
            ForkRequest(
                parent_run_id="P1",
                parent_t=1,
                mutation=MutationPatch(food=999.0),
                n_turns=2,
                reason="test",
            ),
            "stub-key",
            registry,
            logs_root,
            tasks,
        )
        # Drain the background replay so cleanup is clean.
        for task in list(tasks):
            await task
        return response.child_run_id

    assert len(asyncio.run(go())) > 0


def test_create_fork_writes_world_mutation_when_patch_non_empty(
    logs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_invoke(monkeypatch)
    _make_parent_log(logs_root / "2026-05-21" / "race-100000.duckdb", "P2", _state())

    async def go() -> tuple[str, str]:
        registry = LiveRunRegistry()
        tasks: set[asyncio.Task[None]] = set()
        response = await create_fork(
            ForkRequest(
                parent_run_id="P2",
                parent_t=1,
                mutation=MutationPatch(food=42.0),
                n_turns=0,
                reason="mutate food",
            ),
            "stub-key",
            registry,
            logs_root,
            tasks,
        )
        for task in list(tasks):
            await task
        return response.child_run_id, response.db_path

    child_run_id, db_path = asyncio.run(go())
    with duckdb.connect(db_path, read_only=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND kind='world_mutation'",
            [child_run_id],
        ).fetchone()
    assert row is not None and row[0] == 1
