"""The TypeSafe clock publishes advice without joining the actor's latency path."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import Goal, GoalManager
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.policy import evaluate_policy_once, policy_loop
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.advice import PolicyAdvice, PolicyRequest, readonly_probabilities
from gameplay_agent.providers.policy import PolicyAdvisorError

from tests.loop_fakes import FakeActuator, FakeSource

if TYPE_CHECKING:
    from pathlib import Path


class FakeAdvisor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[PolicyRequest] = []
        self.closed = False

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        if self.fail:
            raise PolicyAdvisorError("service unavailable")
        candidate = next(item for item in request.candidates if item.id == "build_house")
        return PolicyAdvice(
            source_tick=request.source_tick,
            source_captured_at=request.source_captured_at,
            created_at=request.source_captured_at,
            model="jev-1.13.0",
            candidate_ids=frozenset(item.id for item in request.candidates),
            action_choice=candidate.id,
            action_confidence=0.9,
            action_probabilities=readonly_probabilities({candidate.id: 0.9}),
            allocation_focus="wood",
            allocation_confidence=0.8,
            allocation_probabilities=readonly_probabilities({"wood": 0.8}),
        )

    async def aclose(self) -> None:
        self.closed = True


def _context(tmp_path: Path) -> LoopContext:
    goals = GoalManager()
    goals.set_goals(
        [
            Goal(
                name="Reach Feudal",
                type="global",
                metric="age",
                target="Feudal Age",
                priority=9,
                created_turn=0,
            )
        ]
    )
    return LoopContext(
        memory=AgentMemory(),
        goal_manager=goals,
        goal_logger=GoalLogger(tmp_path),
        source=FakeSource(),
        actuator=FakeActuator(),
    )


def test_one_evaluation_publishes_frame_tied_advice(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    advisor = FakeAdvisor()
    frame = Perception(tick=8)

    assert asyncio.run(evaluate_policy_once(ctx, advisor, frame))
    published = ctx.policy_advice.latest()

    assert published is not None
    assert published.source_tick == 8
    assert advisor.requests[0].goals[0].name == "Reach Feudal"


def test_expected_provider_failure_preserves_the_rule_fallback(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    advisor = FakeAdvisor(fail=True)

    assert not asyncio.run(evaluate_policy_once(ctx, advisor, Perception(tick=2)))
    assert ctx.policy_advice.latest() is None


def test_policy_loop_skips_alarm_frames_and_closes_provider(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    advisor = FakeAdvisor()
    ctx.frames.put(Perception(tick=1, alarm=True))

    async def drive() -> None:
        task = asyncio.create_task(policy_loop(ctx, advisor, interval_seconds=0.001))
        await asyncio.sleep(0.01)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())

    assert advisor.requests == []
    assert advisor.closed
