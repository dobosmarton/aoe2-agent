"""Deliberate work is reserved for combat, handoff, and bounded recovery."""

from __future__ import annotations

import asyncio
import time

from gameplay_agent.executor import ActionLedger, ActionOutcome
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops import deliberate
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.state import PolicyState
from gameplay_agent.providers.base import LLMResult
from gameplay_agent.providers.strategist import StrategistProvider

from tests.loop_fakes import FakeActuator, FakeSource


class _FakeProvider:
    def __init__(self, *, executed: bool = True) -> None:
        self.executed = executed
        self.planned = 0
        self.acted = 0
        self.recovered = 0

    async def plan(self, _context: str, _width: int = 0, _height: int = 0) -> LLMResult:
        self.planned += 1
        return LLMResult(reasoning="unused", actions=[], observations={})

    async def act(self, _context: str, _width: int = 0, _height: int = 0) -> LLMResult:
        self.acted += 1
        return self._response()

    async def act_recovery(self, _context: str, _width: int = 0, _height: int = 0) -> LLMResult:
        self.recovered += 1
        return self._response()

    def _response(self) -> LLMResult:
        return LLMResult(
            reasoning="bounded tool work",
            actions=[{"type": "press", "key": "q", "intent": "model suggestion"}],
            observations={},
            actions_already_executed=self.executed,
            success_count=1 if self.executed else 0,
        )


def _context(tmp_path) -> LoopContext:
    memory = AgentMemory()
    memory.start_game()
    return LoopContext(
        memory=memory,
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=FakeSource(),
        actuator=FakeActuator(),
        ledger=ActionLedger(),
    )


def _frame(*, alarm: bool = False, food: int = 100) -> Perception:
    return Perception(
        alarm=alarm,
        tick=1,
        world=PolicyState(food=food, known_resources=frozenset({"food"})),
    )


def _trigger(
    ctx: LoopContext, frame: Perception, *, food_age: float = 0.0, cooldown: float = 0.0
) -> deliberate.Trigger | None:
    now = time.monotonic()
    return deliberate._trigger(ctx, frame, now - food_age, now - cooldown if cooldown else 0.0)


def test_quiet_frames_do_not_request_a_discarded_plan(tmp_path) -> None:
    assert _trigger(_context(tmp_path), _frame()) is None


def test_alarm_and_requested_handoff_trigger_deliberate_work(tmp_path) -> None:
    ctx = _context(tmp_path)
    assert _trigger(ctx, _frame(alarm=True)) == "alarm"
    ctx.tactical_requested.set()
    assert _trigger(ctx, _frame()) == "handoff"


def test_population_cap_and_goal_change_are_routine_policy_facts(tmp_path) -> None:
    ctx = _context(tmp_path)
    ctx.goal_manager.set_goals([])
    capped = Perception(world=PolicyState(population=20, population_cap=20))
    assert _trigger(ctx, capped) is None


def test_three_real_failures_trigger_recovery_with_cooldown(tmp_path) -> None:
    ctx = _context(tmp_path)
    assert ctx.ledger is not None
    for _ in range(3):
        ctx.ledger.record_failure("build_farm: failed settlement")
    assert _trigger(ctx, _frame()) == "recovery"
    assert _trigger(ctx, _frame(), cooldown=1.0) is None


def test_famine_needs_thirty_seconds_without_food_progress(tmp_path) -> None:
    ctx = _context(tmp_path)
    assert _trigger(ctx, _frame(food=10), food_age=29.0) is None
    assert _trigger(ctx, _frame(food=10), food_age=31.0) == "recovery"


def test_intentional_wait_and_pending_operations_are_not_failures(tmp_path) -> None:
    ctx = _context(tmp_path)
    assert ctx.ledger is not None
    ctx.ledger.record_outcome(ActionOutcome(1, "build_house", "pending", "awaiting HUD settlement"))
    assert _trigger(ctx, _frame()) is None


def test_alarm_acts_under_the_input_lock_without_planning(tmp_path) -> None:
    ctx = _context(tmp_path)
    held: list[bool] = []

    class _Watcher(_FakeProvider):
        async def act(self, context: str, width: int = 0, height: int = 0) -> LLMResult:
            held.append(ctx.input_lock.locked())
            return await super().act(context, width, height)

    provider = _Watcher()
    asyncio.run(deliberate.deliberate_once(ctx, provider, _frame(alarm=True), 1, "alarm"))
    assert held == [True]
    assert (provider.planned, provider.acted, provider.recovered) == (0, 1, 0)
    assert ctx.memory.turn_count == 1


def test_recovery_uses_its_own_bounded_tool_path(tmp_path) -> None:
    ctx = _context(tmp_path)
    provider = _FakeProvider()
    asyncio.run(deliberate.deliberate_once(ctx, provider, _frame(food=10), 1, "recovery"))
    assert (provider.planned, provider.acted, provider.recovered) == (0, 0, 1)


def test_unexecuted_model_actions_cannot_bypass_catalog(tmp_path) -> None:
    ctx = _context(tmp_path)
    asyncio.run(
        deliberate.deliberate_once(
            ctx, _FakeProvider(executed=False), _frame(alarm=True), 1, "alarm"
        )
    )
    assert ctx.actuator.batches == []


def test_strategist_task_is_cancelled_when_loop_stops(tmp_path, monkeypatch) -> None:
    ctx = _context(tmp_path)
    launched = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang() -> None:
        launched.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(
        deliberate,
        "maybe_launch_strategist",
        lambda *_args: asyncio.create_task(hang()),
    )

    async def drive() -> None:
        task = asyncio.create_task(
            deliberate.deliberate_loop(ctx, StrategistProvider(), _FakeProvider())
        )
        ctx.frames.put(_frame())
        await asyncio.wait_for(launched.wait(), timeout=2.0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=2.0)
        assert cancelled.is_set()

    asyncio.run(drive())
