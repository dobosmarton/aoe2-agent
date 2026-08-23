"""Unit tests for loops/deliberate.py — the LLM, on exceptions only."""

from __future__ import annotations

import asyncio

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import Goal, GoalManager
from gameplay_agent.loops import deliberate
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import STUCK_LOOP_THRESHOLD, AgentMemory
from gameplay_agent.providers.base import LLMResult
from gameplay_agent.providers.strategist import StrategistProvider

from tests.loop_fakes import FakeActuator, FakeSource

_PRESS = {"type": "press", "key": "q", "intent": "queue"}


class _FakeProvider:
    """Records which path was taken and returns a canned turn."""

    def __init__(self, actions: list[dict[str, object]] | None = None, stall: float = 0.0) -> None:
        self.actions = actions or []
        self.stall = stall
        self.planned = 0
        self.acted = 0

    async def plan(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
        self.planned += 1
        if self.stall:
            await asyncio.sleep(self.stall)
        return LLMResult(reasoning="planned", actions=list(self.actions), observations={})

    async def act(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
        self.acted += 1
        return LLMResult(
            reasoning="acted",
            actions=list(self.actions),
            observations={},
            actions_already_executed=True,
            success_count=len(self.actions),
        )


@pytest.fixture
def gates():
    ex.reset_build_gates()
    yield
    ex.reset_build_gates()


def _context(tmp_path) -> LoopContext:
    memory = AgentMemory()
    memory.start_game()
    return LoopContext(
        memory=memory,
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=FakeSource(),
        actuator=FakeActuator(),
    )


def _frame(alarm: bool = False) -> Perception:
    return Perception(alarm=alarm, tick=1)


def _goal(name: str) -> Goal:
    """A goal whose only job is to move the name set."""
    return Goal(
        name=name, type="global", metric="age", target="Feudal Age", priority=5, created_turn=0
    )


# ---------------------------------------------------------------------------
# The triggers
# ---------------------------------------------------------------------------


def test_a_quiet_frame_asks_nothing(tmp_path, gates, monkeypatch) -> None:
    monkeypatch.setattr(deliberate.config, "deliberate_interval", 10)
    assert deliberate._trigger(_context(tmp_path), _frame(), tick=1) is None


def test_an_alarm_triggers(tmp_path, gates) -> None:
    assert deliberate._trigger(_context(tmp_path), _frame(alarm=True), tick=1) == "alarm"


def test_a_pop_cap_triggers(tmp_path, gates) -> None:
    ex.observe_hud(20, 20, {"wood": 200})  # housed: 20/20
    assert deliberate._trigger(_context(tmp_path), _frame(), tick=1) == "housed"


def test_a_stuck_streak_triggers(tmp_path, gates) -> None:
    ctx = _context(tmp_path)
    for _ in range(STUCK_LOOP_THRESHOLD):
        turn = ctx.memory.create_turn(reasoning="x", actions=[])
        turn.verification = "- no visible change: build produced no new building"
    assert deliberate._trigger(ctx, _frame(), tick=1) == "stuck"


def test_a_goal_change_triggers(tmp_path, gates) -> None:
    ctx = _context(tmp_path)
    ctx.goal_manager.set_goals([_goal("Hold the hill")])
    assert deliberate._trigger(ctx, _frame(), tick=1) == "goals"


def test_a_goal_change_triggers_only_once(tmp_path, gates, monkeypatch) -> None:
    monkeypatch.setattr(deliberate.config, "deliberate_interval", 10)
    ctx = _context(tmp_path)
    ctx.goal_manager.set_goals([_goal("Hold the hill")])
    deliberate._trigger(ctx, _frame(), tick=1)
    assert deliberate._trigger(ctx, _frame(), tick=1) is None


def test_famine_triggers(tmp_path, gates) -> None:
    ctx = _context(tmp_path)
    ctx.memory.game_state.resources["food"] = 10
    assert deliberate._trigger(ctx, _frame(), tick=1) == "famine"


def test_the_nth_frame_triggers(tmp_path, gates, monkeypatch) -> None:
    """The sanity check: it is the only game-over signal the agent has."""
    monkeypatch.setattr(deliberate.config, "deliberate_interval", 10)
    assert deliberate._trigger(_context(tmp_path), _frame(), tick=10) == "interval"


# ---------------------------------------------------------------------------
# Routine ticks plan; exceptions act
# ---------------------------------------------------------------------------


def test_a_routine_tick_plans(tmp_path, gates) -> None:
    provider = _FakeProvider()
    asyncio.run(deliberate.deliberate_once(_context(tmp_path), provider, _frame(), 1, "interval"))
    assert (provider.planned, provider.acted) == (1, 0)


def test_a_routine_tick_discards_the_llm_actions(tmp_path, gates) -> None:
    """The rules own routine play. The LLM's clicks are logged, never pressed."""
    ctx = _context(tmp_path)
    provider = _FakeProvider(actions=[_PRESS, _PRESS, _PRESS])
    asyncio.run(deliberate.deliberate_once(ctx, provider, _frame(), 1, "interval"))
    assert ctx.actuator.batches == []


def test_an_alarm_takes_the_acting_path(tmp_path, gates) -> None:
    provider = _FakeProvider()
    asyncio.run(deliberate.deliberate_once(_context(tmp_path), provider, _frame(True), 1, "alarm"))
    assert (provider.planned, provider.acted) == (0, 1)


def test_the_acting_path_holds_the_input_lock(tmp_path, gates) -> None:
    """Two loops must never type at once."""
    ctx = _context(tmp_path)
    held: list[bool] = []

    class _Watcher(_FakeProvider):
        async def act(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
            held.append(ctx.input_lock.locked())
            return await super().act(context, width, height)

    asyncio.run(deliberate.deliberate_once(ctx, _Watcher(), _frame(True), 1, "alarm"))
    assert held == [True]


def test_a_routine_tick_holds_no_lock(tmp_path, gates) -> None:
    """`plan` cannot act, so blocking the act loop for it would be waste."""
    ctx = _context(tmp_path)
    held: list[bool] = []

    class _Watcher(_FakeProvider):
        async def plan(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
            held.append(ctx.input_lock.locked())
            return await super().plan(context, width, height)

    asyncio.run(deliberate.deliberate_once(ctx, _Watcher(), _frame(), 1, "interval"))
    assert held == [False]


# ---------------------------------------------------------------------------
# Bookkeeping the rest of the run depends on
# ---------------------------------------------------------------------------


def test_a_turn_is_recorded(tmp_path, gates) -> None:
    """`turn_count` gates the saved trace and the post-game rule extraction."""
    ctx = _context(tmp_path)
    asyncio.run(deliberate.deliberate_once(ctx, _FakeProvider(), _frame(), 1, "interval"))
    assert ctx.memory.turn_count == 1


def test_victory_ends_the_run(tmp_path, gates) -> None:
    ctx = _context(tmp_path)

    class _Winner(_FakeProvider):
        async def plan(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
            return LLMResult(reasoning="won", actions=[], observations={"game_state": "victory"})

    asyncio.run(deliberate.deliberate_once(ctx, _Winner(), _frame(), 1, "interval"))
    assert ctx.memory.game_end_reason == "victory"


def test_a_tick_records_its_own_latency(tmp_path, gates) -> None:
    ctx = _context(tmp_path)
    asyncio.run(deliberate.deliberate_once(ctx, _FakeProvider(), _frame(), 1, "interval"))
    assert "deliberate" in ctx.latency.snapshot().loops


# ---------------------------------------------------------------------------
# The whole point: a slow LLM must not stop the agent acting
# ---------------------------------------------------------------------------


def test_the_loop_leaves_when_the_game_ends(tmp_path, gates) -> None:
    """No frame will ever arrive, so the loop must poll the stop flag."""
    ctx = _context(tmp_path)
    strategist = StrategistProvider()

    async def drive() -> None:
        ctx.request_stop("interrupted")
        loop = deliberate.deliberate_loop(ctx, strategist, _FakeProvider())
        await asyncio.wait_for(loop, timeout=2.0)

    asyncio.run(drive())
