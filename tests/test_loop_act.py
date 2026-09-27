"""Unit tests for loops/act.py — the clock that decides and presses keys."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import Goal, GoalManager
from gameplay_agent.loops import act, perceive, source
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception, SpatialRefresh
from gameplay_agent.loops.source import frame_refresh
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.advice import PolicyAdvice, PolicyRequest, readonly_probabilities
from gameplay_agent.policy.state import PolicyState
from gameplay_agent.providers.policy import PolicyAdvisorError
from gameplay_agent.turn_timing import ACT_LOOP

from tests.factories import make_entity as _ent
from tests.loop_fakes import FakeActuator, FakePolicyAdvisor, FakeSource

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
    """A frame with a known idle worker and visible food."""
    return Perception(
        entities=(_ent("town_center", (0.0, 0.0)), _ent("sheep", (10.0, 10.0))),
        world=PolicyState(
            food=100,
            wood=100,
            population=22,
            population_cap=30,
            villagers_ordered=30,
            buildings_seen=frozenset({"mill", "lumber_camp"}),
            idle_present=True,
        ),
        tick=tick,
    )


def _house_frame(tick: int = 1) -> Perception:
    return Perception(
        tick=tick,
        world=PolicyState(wood=100, population=4, population_cap=5, idle_present=True),
    )


def _advice(
    request: PolicyRequest,
    *,
    action: str = "build_house",
    confidence: float = 0.9,
) -> PolicyAdvice:
    return PolicyAdvice(
        source_tick=request.source_tick,
        source_captured_at=request.source_captured_at,
        model="jev-1.13.0",
        action_choice=action,  # pyright: ignore[reportArgumentType]
        action_confidence=confidence,
        action_probabilities=readonly_probabilities({action: confidence}),
        allocation_focus="wood",
        allocation_confidence=0.9,
        allocation_probabilities=readonly_probabilities({"wood": 0.9}),
    )


class _HouseAdvisor:
    def __init__(
        self, *, confidence: float = 0.9, fail: bool = False, action: str = "build_house"
    ) -> None:
        self.confidence = confidence
        self.fail = fail
        self.action = action
        self.requests: list[PolicyRequest] = []
        self.closed = False

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        if self.fail:
            raise PolicyAdvisorError("service unavailable")
        return _advice(request, action=self.action, confidence=self.confidence)

    async def aclose(self) -> None:
        self.closed = True


class _WaitingAdvisor(_HouseAdvisor):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return _advice(request)


class _NeverAdvisor(_HouseAdvisor):
    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _WrongFrameAdvisor(_HouseAdvisor):
    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        return replace(_advice(request), source_tick=request.source_tick + 1)


def test_a_decision_reaches_the_actuator(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _run(act.act_once(ctx, _HouseAdvisor(action="assign_food"), _idle_frame(), tick=1))
    assert [a["type"] for a in actuator.actions] == ["assign_idle"]


def test_a_tick_records_its_own_latency(tmp_path, gates) -> None:
    """`loop_arch` flips to "clocks" on the presence of an act tick."""
    ctx = _context(tmp_path)
    _run(act.act_once(ctx, FakePolicyAdvisor(), _idle_frame(), tick=1))
    assert ACT_LOOP in ctx.latency.snapshot().loops


def test_the_decide_phase_is_timed_apart_from_the_execute_phase(tmp_path, gates) -> None:
    """The bounded policy wait is measured apart from input execution."""
    ctx = _context(tmp_path)
    _run(act.act_once(ctx, FakePolicyAdvisor(), _idle_frame(), tick=1))
    assert set(ctx.latency.snapshot().of(ACT_LOOP).phase_p50_ms) == {"policy", "execute"}


def test_an_empty_decision_presses_nothing(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    _run(act.act_once(ctx, FakePolicyAdvisor(), Perception(), tick=1))
    assert actuator.batches == []


def test_an_alarm_frame_leaves_combat_to_the_llm(tmp_path, gates) -> None:
    """`decide` returns nothing under alarm — the rules do not fight."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    frame = Perception(entities=_idle_frame().entities, alarm=True)
    advisor = FakePolicyAdvisor()
    _run(act.act_once(ctx, advisor, frame, tick=1))
    assert actuator.batches == []
    assert advisor.requests == []


def test_a_held_input_lock_skips_the_tick(tmp_path, gates) -> None:
    """The combat tool loop is typing; queueing behind it would act on a frame
    the burst has already invalidated."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)

    async def drive() -> None:
        async with ctx.input_lock:
            await act.act_once(ctx, FakePolicyAdvisor(), _idle_frame(), tick=1)

    _run(drive())
    assert actuator.batches == []


def test_a_skipped_tick_is_not_measured(tmp_path, gates) -> None:
    """A combat burst must not inflate the act p95."""
    ctx = _context(tmp_path)

    async def drive() -> None:
        async with ctx.input_lock:
            await act.act_once(ctx, FakePolicyAdvisor(), _idle_frame(), tick=1)

    _run(drive())
    assert ACT_LOOP not in ctx.latency.snapshot().loops


def test_results_reach_the_action_ledger(tmp_path, gates) -> None:
    """One named action contributes one executed action result."""
    ctx = _context(tmp_path)
    _run(act.act_once(ctx, _HouseAdvisor(action="assign_food"), _idle_frame(), tick=1))
    assert ctx.memory.executed_actions == 1


def test_the_loop_decides_once_per_frame(tmp_path, gates) -> None:
    """Every rule guard reads the HUD, and the HUD only moves on a new frame."""
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)

    async def drive() -> None:
        task = asyncio.create_task(act.act_loop(ctx, _HouseAdvisor(action="assign_food")))
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

    async def drive() -> None:
        task = asyncio.create_task(act.act_loop(ctx, _HouseAdvisor(action="assign_food")))
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

    advisor = FakePolicyAdvisor()

    async def drive() -> None:
        ctx.request_stop("interrupted")
        await asyncio.wait_for(act.act_loop(ctx, advisor), timeout=2.0)

    _run(drive())  # must not hang
    assert advisor.closed


def test_actor_waits_for_and_executes_same_frame_advice(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    frame = _house_frame()
    advisor = _WaitingAdvisor()
    batches_while_waiting: list[int] = []

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, frame, tick=1))
        await advisor.started.wait()
        batches_while_waiting.append(len(actuator.batches))
        advisor.release.set()
        await task

    _run(drive())

    assert (
        batches_while_waiting,
        [action["type"] for action in actuator.actions],
        advisor.requests[0].source_tick,
    ) == ([0], ["build"], frame.tick)


def test_provider_failure_uses_same_frame_fallback(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)

    _run(act.act_once(ctx, _HouseAdvisor(fail=True), _idle_frame(), tick=1))

    assert [action["type"] for action in actuator.actions] == ["assign_idle"]


def test_advice_for_a_different_frame_uses_same_frame_fallback(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)

    _run(act.act_once(ctx, _WrongFrameAdvisor(), _idle_frame(), tick=1))

    assert [action["type"] for action in actuator.actions] == ["assign_idle"]


def test_low_confidence_advice_uses_same_frame_fallback(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)

    _run(act.act_once(ctx, _HouseAdvisor(confidence=0.1), _idle_frame(), tick=1))

    assert [action["type"] for action in actuator.actions] == ["assign_idle"]


def test_policy_timeout_uses_same_frame_fallback(tmp_path, gates, monkeypatch) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    monkeypatch.setattr(act.config, "policy_timeout", 0.001)

    _run(act.act_once(ctx, _NeverAdvisor(), _idle_frame(), tick=1))

    assert [action["type"] for action in actuator.actions] == ["assign_idle"]


def test_newer_compatible_frame_revalidates_in_flight_advice(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()
    frame = _house_frame()

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, frame, tick=1))
        await advisor.started.wait()
        ctx.frames.put(_house_frame(tick=2))
        advisor.release.set()
        await task

    _run(drive())
    assert [action["type"] for action in actuator.actions] == ["build"]


def test_changed_input_revision_rejects_old_advice_and_uses_latest_fallback(
    tmp_path, gates
) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, _house_frame(), tick=1))
        await advisor.started.wait()
        ctx.frames.put(
            Perception(
                tick=2,
                input_revision=1,
                world=PolicyState(wood=100, population=4, population_cap=30, idle_present=True),
            )
        )
        advisor.release.set()
        await task

    _run(drive())
    assert actuator.actions[0]["building_key"] == "w"  # build_mill, not the old house advice


def test_goal_revision_change_rejects_advice(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, _house_frame(), tick=1))
        await advisor.started.wait()
        ctx.goal_manager.set_goals(
            [
                Goal(
                    name="Reach Feudal",
                    type="global",
                    metric="age",
                    target="Feudal Age",
                    priority=8,
                    created_turn=1,
                )
            ]
        )
        ctx.frames.put(
            Perception(
                tick=2,
                world=PolicyState(wood=100, population=4, population_cap=30, idle_present=True),
            )
        )
        advisor.release.set()
        await task

    _run(drive())
    assert actuator.actions[0]["building_key"] == "w"


def test_alarm_arriving_during_policy_wait_prevents_fallback(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, _house_frame(), tick=1))
        await advisor.started.wait()
        ctx.frames.put(replace(_house_frame(tick=2), alarm=True))
        advisor.release.set()
        await task

    _run(drive())
    assert actuator.batches == []


def test_revalidated_execution_consumes_latest_frame_once(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()

    async def drive() -> None:
        task = asyncio.create_task(act.act_loop(ctx, advisor))
        ctx.frames.put(_house_frame())
        await advisor.started.wait()
        ctx.frames.put(_house_frame(tick=2))
        advisor.release.set()
        for _ in range(20):
            await asyncio.sleep(0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=2.0)

    _run(drive())
    assert len(advisor.requests) == 1
    assert len(actuator.batches) == 1


def test_input_lock_supersedes_in_flight_advice(tmp_path, gates) -> None:
    actuator = FakeActuator()
    ctx = _context(tmp_path, actuator)
    advisor = _WaitingAdvisor()

    async def drive() -> None:
        task = asyncio.create_task(act.act_once(ctx, advisor, Perception(tick=1), tick=1))
        await advisor.started.wait()
        async with ctx.input_lock:
            advisor.release.set()
            await task

    _run(drive())
    assert actuator.batches == []


# ---------------------------------------------------------------------------
# The refresh hook — what keeps detection off the act task
# ---------------------------------------------------------------------------


def test_the_refresh_hook_waits_for_spatial_capture(tmp_path, gates) -> None:
    """A composite action waits for a post-input spatial capture."""
    ctx = _context(tmp_path)
    refresh = frame_refresh(ctx.frames)
    ctx.frames.put(_idle_frame(tick=1))

    async def drive() -> bool:
        waiting = asyncio.create_task(refresh())
        await asyncio.sleep(0)
        request = ctx.frames.pending_spatial_refresh()
        assert request is not None
        ctx.frames.complete_spatial_refresh(
            request,
            SpatialRefresh(captured_at=time.monotonic(), input_revision=0, spatial_valid=True),
        )
        return await asyncio.wait_for(waiting, timeout=2.0)

    assert _run(drive()) is True


def test_urgent_spatial_refresh_bypasses_the_next_slow_full_pass(
    tmp_path, gates, monkeypatch
) -> None:
    """The camera can be refreshed while full OCR would exceed the deadline."""

    class SlowSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.full_started = asyncio.Event()
            self.release_full = asyncio.Event()
            self.full_calls = 0
            self.spatial_captures = 0

        async def capture(self, tick, timings):
            self.full_calls += 1
            if self.full_calls == 1:
                self.full_started.set()
                await self.release_full.wait()
            else:
                await asyncio.Event().wait()
            return await super().capture(tick, timings)

        async def capture_spatial(self, timings):
            self.spatial_captures += 1
            return SpatialRefresh(
                captured_at=time.monotonic(), input_revision=0, spatial_valid=True
            )

    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: True)
    monkeypatch.setattr(source, "_REFRESH_TIMEOUT", 0.1)
    scripted = SlowSource()
    ctx = LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=scripted,
        actuator=FakeActuator(),
    )

    async def drive() -> bool:
        perceive_task = asyncio.create_task(perceive.perceive_loop(ctx))
        try:
            await asyncio.wait_for(scripted.full_started.wait(), timeout=1.0)
            refresh_task = asyncio.create_task(frame_refresh(ctx.frames)())
            await asyncio.sleep(0)
            scripted.release_full.set()
            return await asyncio.wait_for(refresh_task, timeout=1.0)
        finally:
            perceive_task.cancel()
            await asyncio.gather(perceive_task, return_exceptions=True)

    assert _run(drive()) is True
    assert scripted.spatial_captures == 1
    assert scripted.captures == 1


def test_the_refresh_hook_gives_up_after_a_timeout(tmp_path, gates, monkeypatch) -> None:
    """A hung perceive loop must not freeze the act loop. The outer wait turns a
    regression into a failure instead of a hung suite."""
    monkeypatch.setattr(source, "_REFRESH_TIMEOUT", 0.01)
    ctx = _context(tmp_path)
    assert _run(asyncio.wait_for(frame_refresh(ctx.frames)(), timeout=2.0)) is False


def test_spatial_refresh_rejects_a_capture_from_before_the_input(tmp_path, gates) -> None:
    ctx = _context(tmp_path)

    async def drive() -> bool:
        waiting = asyncio.create_task(frame_refresh(ctx.frames)())
        await asyncio.sleep(0)
        request = ctx.frames.pending_spatial_refresh()
        assert request is not None
        ctx.frames.complete_spatial_refresh(
            request, SpatialRefresh(captured_at=0.0, input_revision=0, spatial_valid=True)
        )
        return await waiting

    assert _run(drive()) is False


def test_a_late_refresh_cannot_satisfy_the_next_request(tmp_path, gates, monkeypatch) -> None:
    ctx = _context(tmp_path)

    async def drive() -> bool:
        monkeypatch.setattr(source, "_REFRESH_TIMEOUT", 0.01)
        first = asyncio.create_task(frame_refresh(ctx.frames)())
        await asyncio.sleep(0)
        old_request = ctx.frames.pending_spatial_refresh()
        assert old_request is not None
        assert await first is False

        monkeypatch.setattr(source, "_REFRESH_TIMEOUT", 1.0)
        second = asyncio.create_task(frame_refresh(ctx.frames)())
        await asyncio.sleep(0)
        new_request = ctx.frames.pending_spatial_refresh()
        assert new_request is not None
        ctx.frames.complete_spatial_refresh(
            old_request,
            SpatialRefresh(captured_at=time.monotonic(), input_revision=0, spatial_valid=True),
        )
        assert not second.done()
        ctx.frames.complete_spatial_refresh(
            new_request,
            SpatialRefresh(captured_at=time.monotonic(), input_revision=0, spatial_valid=True),
        )
        return await second

    assert _run(drive()) is True
