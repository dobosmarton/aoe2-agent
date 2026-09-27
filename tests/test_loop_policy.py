"""Frame-local policy request construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import Goal, GoalManager
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.request import policy_request

from tests.loop_fakes import FakeActuator, FakeSource

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_build_gates() -> None:
    ex.reset_build_gates()


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


def test_request_is_tied_to_one_frame_and_its_goals(tmp_path: Path) -> None:
    request = policy_request(_context(tmp_path), Perception(tick=8, captured_at=12.5))

    assert request.source_tick == 8
    assert request.source_captured_at == 12.5
    assert request.goals[0].name == "Reach Feudal"


def test_pending_house_is_part_of_policy_state_and_not_a_candidate(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.memory.game_state.population = 4
    ctx.memory.game_state.population_cap = 5
    ctx.memory.game_state.resources["wood"] = 200
    ex.observe_hud(4, 5, {"wood": 200})
    ex._note_pending_placement("q")

    request = policy_request(ctx, Perception(tick=1))

    assert request.state.pending_buildings == frozenset({"house"})
    assert "build_house" not in {candidate.id for candidate in request.candidates}
