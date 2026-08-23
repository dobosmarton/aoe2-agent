"""Unit tests for loops/perceive.py — the clock that reads the world."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops import perceive
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.turn_timing import PERCEIVE_LOOP

from tests.factories import make_entity as _ent
from tests.loop_fakes import FakeActuator, FakeSource

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


@pytest.fixture
def gates():
    ex.reset_build_gates()
    yield
    ex.reset_build_gates()


def _context(tmp_path, source: FakeSource) -> LoopContext:
    return LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=source,
        actuator=FakeActuator(),
    )


def test_a_pass_publishes_a_frame(tmp_path, gates) -> None:
    ctx = _context(tmp_path, FakeSource([Perception(width=800, height=600)]))
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.width == 800


def test_the_frame_id_is_the_pass_number(tmp_path, gates) -> None:
    """`act_decided` names the frame it acted on, so the id must be the pipe's."""
    ctx = _context(tmp_path, FakeSource([Perception(), Perception()]))
    _run(perceive.perceive_once(ctx, tick=1))
    _run(perceive.perceive_once(ctx, tick=2))
    frame = ctx.frames.latest()
    assert frame is not None and frame.tick == 2


def test_a_pass_records_its_own_latency(tmp_path, gates) -> None:
    """`loop_arch` and the perceive budget both read this recorder."""
    ctx = _context(tmp_path, FakeSource())
    _run(perceive.perceive_once(ctx, tick=1))
    assert PERCEIVE_LOOP in ctx.latency.snapshot().loops


def _read_hud(tmp_path) -> LoopContext:
    """One pass over a frame carrying a HUD reading."""
    ctx = _context(
        tmp_path,
        FakeSource([Perception(hud_readings={"food": 120, "wood": 250, "population": "12/20"})]),
    )
    _run(perceive.perceive_once(ctx, tick=1))
    return ctx


def test_a_hud_reading_reaches_the_game_state(tmp_path, gates) -> None:
    assert _read_hud(tmp_path).memory.game_state.resources.get("wood") == 250


def test_a_hud_reading_reaches_the_build_gates(tmp_path, gates) -> None:
    """The perceive loop is the only writer of the gates' HUD snapshot."""
    ctx = _read_hud(tmp_path)
    assert ex._build_gates.resources == ctx.memory.game_state.resources


def test_no_entities_means_no_alarm(tmp_path, gates) -> None:
    """An empty frame must not cost an alarm check — it cannot find a threat."""
    ctx = _context(tmp_path, FakeSource([Perception()]))
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.alarm is False


def test_the_alarm_rides_on_the_frame(tmp_path, gates) -> None:
    """Act reads the alarm off the frame, so it must be stamped there, not on
    a variable only the old single tick could see."""
    # 3 is the alarm floor: one stray spearman once rang the town bell and
    # garrisoned the whole economy (exp_0013, turn 14).
    threats = tuple(_ent("knight_line", (100.0, 100.0), f"knight_{i}") for i in range(3))
    owner = _enemy_owner()
    ctx = _context(
        tmp_path,
        FakeSource(
            [Perception(entities=threats)],
            ownership={f"knight_{i}": (owner, 0.9) for i in range(3)},
        ),
    )
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.alarm is True


def test_the_loop_stops_when_the_game_is_gone(tmp_path, gates, monkeypatch) -> None:
    """Perceive is the loop that can see the window close."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: False)
    ctx = _context(tmp_path, FakeSource())
    _run(perceive.perceive_loop(ctx))
    assert ctx.memory.game_end_reason == "game_not_found"


def _unfocusable(tmp_path, monkeypatch) -> tuple[LoopContext, FakeSource]:
    """A window that never focuses, with the retry wait removed."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: False)
    monkeypatch.setattr(perceive, "_MAX_FOCUS_FAILURES", 3)
    monkeypatch.setattr(perceive, "_FOCUS_RETRY_DELAY", 0.0)
    source = FakeSource()
    ctx = _context(tmp_path, source)
    _run(perceive.perceive_loop(ctx))
    return ctx, source


def test_permanent_focus_loss_ends_the_run_labelled(tmp_path, gates, monkeypatch) -> None:
    """F-1: 12 of 30 iterations went to an unfocusable window, unlabelled."""
    ctx, _source = _unfocusable(tmp_path, monkeypatch)
    assert ctx.memory.game_end_reason == "lost_focus"


def test_an_unplayable_window_is_never_billed_a_frame(tmp_path, gates, monkeypatch) -> None:
    _ctx, source = _unfocusable(tmp_path, monkeypatch)
    assert source.captures == 0


def test_the_loop_resumes_when_focus_returns(tmp_path, gates, monkeypatch) -> None:
    """A transient focus loss must not end the run — it costs frames, not the game."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive.config, "perceive_interval", 0.0)
    focus = iter([False, False, True])
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: next(focus, True))
    monkeypatch.setattr(perceive, "_FOCUS_RETRY_DELAY", 0.0)
    source = FakeSource()
    ctx = _context(tmp_path, source)

    async def drive() -> None:
        task = asyncio.create_task(perceive.perceive_loop(ctx))
        while source.captures < 2:
            await asyncio.sleep(0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=1.0)

    _run(drive())
    assert source.captures >= 2  # the loop resumed once focus came back


def _enemy_owner():
    """The classifier's enemy label — an ownership map keys the alarm on it."""
    from detection.inference.ownership import Owner

    return Owner.ENEMY
