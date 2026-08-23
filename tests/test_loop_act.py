"""Unit tests for loops/act.py — the clock that decides and presses keys."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops import act, source
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.loops.source import frame_refresh
from gameplay_agent.memory import AgentMemory
from gameplay_agent.turn_timing import ACT_LOOP

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


def _context(tmp_path, actuator: FakeActuator | None = None) -> LoopContext:
    return LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=FakeSource(),
        actuator=actuator if actuator is not None else FakeActuator(),
    )


def _idle_frame(tick: int = 1) -> Perception:
    """A frame the idle rule answers: a lit badge and a sheep to send them to."""
    return Perception(
        entities=(_ent("town_center", (0.0, 0.0)), _ent("sheep", (10.0, 10.0))),
        tick=tick,
    )


def _only_idle_fires(ctx: LoopContext) -> None:
    """A state no build or queue rule answers, so a decision is the idle pair.

    The order ledger is past the Dark-Age target, the prep buildings stand, and
    there is cap headroom — the 3 rules that would otherwise compete.
    """
    state = ctx.memory.game_state
    state.population = 22
    state.population_cap = 30
    state.villagers_ordered = 30
    state.buildings_seen = frozenset({"mill", "lumber_camp"})
    state.resources = {"food": 100, "wood": 100, "gold": 0, "stone": 0}
    state.idle_present = True


def test_a_decision_reaches_the_actuator(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)
    _run(act.act_once(ctx, _idle_frame(), tick=1))
    assert [a["type"] for a in actuator.actions] == ["press", "right_click"]


def test_a_tick_records_its_own_latency(tmp_path, gates) -> None:
    """`loop_arch` flips to "clocks" on the presence of an act tick."""
    ctx = _context(tmp_path)
    _only_idle_fires(ctx)
    _run(act.act_once(ctx, _idle_frame(), tick=1))
    assert ACT_LOOP in ctx.latency.snapshot().loops


def test_the_decide_phase_is_timed_apart_from_the_execute_phase(tmp_path, gates) -> None:
    """The 100 ms budget is asserted on `decide`, which is input-free."""
    ctx = _context(tmp_path)
    _only_idle_fires(ctx)
    _run(act.act_once(ctx, _idle_frame(), tick=1))
    assert set(ctx.latency.snapshot().of(ACT_LOOP).phase_p50_ms) == {"decide", "execute"}


def test_an_empty_decision_presses_nothing(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)
    ctx.memory.game_state.idle_present = False  # nobody idle, nothing else to do
    _run(act.act_once(ctx, Perception(), tick=1))
    assert actuator.batches == []


def test_an_alarm_frame_leaves_combat_to_the_llm(tmp_path, gates) -> None:
    """`decide` returns nothing under alarm — the rules do not fight."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)
    frame = Perception(entities=_idle_frame().entities, alarm=True)
    _run(act.act_once(ctx, frame, tick=1))
    assert actuator.batches == []


def test_a_held_input_lock_skips_the_tick(tmp_path, gates) -> None:
    """The combat tool loop is typing; queueing behind it would act on a frame
    the burst has already invalidated."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)

    async def drive() -> None:
        async with ctx.input_lock:
            await act.act_once(ctx, _idle_frame(), tick=1)

    _run(drive())
    assert actuator.batches == []


def test_a_skipped_tick_is_not_measured(tmp_path, gates) -> None:
    """A combat burst must not inflate the act p95."""
    ctx = _context(tmp_path)
    _only_idle_fires(ctx)

    async def drive() -> None:
        async with ctx.input_lock:
            await act.act_once(ctx, _idle_frame(), tick=1)

    _run(drive())
    assert ACT_LOOP not in ctx.latency.snapshot().loops


def test_results_reach_the_action_ledger(tmp_path, gates) -> None:
    """The rules do most of the acting now, so they must feed action_success."""
    ctx = _context(tmp_path)
    _only_idle_fires(ctx)
    _run(act.act_once(ctx, _idle_frame(), tick=1))
    assert ctx.memory.executed_actions == 2


def test_the_loop_decides_once_per_frame(tmp_path, gates) -> None:
    """Every rule guard reads the HUD, and the HUD only moves on a new frame."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)

    async def drive() -> None:
        task = asyncio.create_task(act.act_loop(ctx))
        ctx.frames.put(_idle_frame(tick=1))
        for _ in range(20):
            await asyncio.sleep(0)  # plenty of turns for a second decision
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=2.0)

    _run(drive())
    assert len(actuator.batches) == 1


def test_the_loop_decides_again_on_the_next_frame(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _only_idle_fires(ctx)

    async def drive() -> None:
        task = asyncio.create_task(act.act_loop(ctx))
        for tick in (1, 2):
            ctx.frames.put(_idle_frame(tick=tick))
            for _ in range(10):
                await asyncio.sleep(0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=2.0)

    _run(drive())
    assert len(actuator.batches) == 2


def test_the_loop_leaves_when_the_game_ends(tmp_path, gates) -> None:
    """No frame will ever arrive, so the loop must poll the stop flag."""
    ctx = _context(tmp_path)

    async def drive() -> None:
        ctx.request_stop("interrupted")
        await asyncio.wait_for(act.act_loop(ctx), timeout=2.0)

    _run(drive())  # must not hang


# ---------------------------------------------------------------------------
# The refresh hook — what keeps detection off the act task
# ---------------------------------------------------------------------------


def test_the_refresh_hook_waits_for_the_next_frame(tmp_path, gates) -> None:
    """A composite action rescans from inside `execute_action`; the hook turns
    that into a wait on the perceive loop instead of a detection."""
    ctx = _context(tmp_path)
    refresh = frame_refresh(ctx.frames)
    ctx.frames.put(_idle_frame(tick=1))

    async def drive() -> int:
        waiting = asyncio.create_task(refresh())
        await asyncio.sleep(0)
        ctx.frames.put(_idle_frame(tick=2))
        await asyncio.wait_for(waiting, timeout=2.0)
        frame = ctx.frames.latest()
        return frame.tick if frame else 0

    assert _run(drive()) == 2


def test_the_refresh_hook_gives_up_after_a_timeout(tmp_path, gates, monkeypatch) -> None:
    """A hung perceive loop must not freeze the act loop. The outer wait turns a
    regression into a failure instead of a hung suite."""
    monkeypatch.setattr(source, "_REFRESH_TIMEOUT", 0.01)
    ctx = _context(tmp_path)
    _run(asyncio.wait_for(frame_refresh(ctx.frames)(), timeout=2.0))
    assert ctx.frames.latest() is None  # it gave up; no frame ever arrived
