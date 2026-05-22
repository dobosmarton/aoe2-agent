"""Tests for arena/web/forks.py (Phase 9, broker-wired Phase 2). Offline.

Broker lifecycle invariants live in `tests/test_event_broker.py`; this
file exercises the fork-specific orchestration (parent resolution,
mutation events, persister-flush-before-return).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import duckdb
import pytest
from pydantic import ValidationError

from arena.web import forks as forks_module
from arena.web.forks import ForkRequest, MutationPatch, create_fork
from evaluation.event_broker import InProcessEventBroker, RunId
from evaluation.event_log import (
    DuckDBEventSink,
    Event,
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
# create_fork
# ---------------------------------------------------------------------------


def test_create_fork_raises_when_parent_run_missing(logs_root: Path) -> None:
    request = ForkRequest(parent_run_id="ghost", parent_t=1)
    with pytest.raises(FileNotFoundError):
        asyncio.run(create_fork(request, "stub-key", InProcessEventBroker(), logs_root, set()))


def test_create_fork_returns_child_run_id(
    logs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_invoke(monkeypatch)
    _make_parent_log(logs_root / "2026-05-21" / "race-100000.duckdb", "P1", _state())

    async def go() -> str:
        broker = InProcessEventBroker()
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
            broker,
            logs_root,
            tasks,
        )
        # Drain the background replay + persister so cleanup is clean.
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
        broker = InProcessEventBroker()
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
            broker,
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


def test_create_fork_closes_broker_run_after_replay(
    logs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broker invariant: after the replay task completes, broker.is_open
    is False AND the DuckDB file is fully written (persister awaited)."""
    _patch_invoke(monkeypatch)
    _make_parent_log(logs_root / "2026-05-21" / "race-100000.duckdb", "P3", _state())

    async def go() -> tuple[bool, int, str]:
        broker = InProcessEventBroker()
        tasks: set[asyncio.Task[None]] = set()
        response = await create_fork(
            ForkRequest(parent_run_id="P3", parent_t=1, n_turns=1),
            "stub-key",
            broker,
            logs_root,
            tasks,
        )
        for task in list(tasks):
            await task

        with duckdb.connect(response.db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE run_id=?",
                [response.child_run_id],
            ).fetchone()
        assert row is not None
        return (
            broker.is_open(RunId(response.child_run_id)),
            int(cast("int", row[0])),
            response.child_run_id,
        )

    is_open, row_count, child_run_id = asyncio.run(go())
    assert is_open is False, "broker should close the run after replay"
    assert row_count >= 1, f"DuckDB should have ≥1 event for {child_run_id} after persister flush"
