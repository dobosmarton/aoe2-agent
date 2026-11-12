"""A strategist run that fails must not cost the agent the goals it has."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import Goal, GoalManager
from gameplay_agent.memory import AgentMemory
from gameplay_agent.strategist_phase import StrategistView, maybe_launch_strategist

if TYPE_CHECKING:
    from pathlib import Path


class _FailingStrategist:
    last_allocation = None

    def should_run(self, **_due: object) -> bool:
        return True

    async def generate_goals(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("model unavailable")


async def _launch_and_wait(goals: GoalManager, log_dir: Path) -> None:
    view = StrategistView(alarm=False, entity_summary="", hud_readings={}, known_buildings="")
    task = maybe_launch_strategist(
        _FailingStrategist(),  # pyright: ignore[reportArgumentType]
        1,
        view,
        AgentMemory(),
        goals,
        GoalLogger(log_dir),
        None,
    )
    if task is not None:
        await task


def test_a_failed_run_keeps_the_current_goals(tmp_path: Path) -> None:
    goals = GoalManager()
    plan = Goal(
        name="Bank food", type="local", metric="food", target=500, priority=9, created_turn=0
    )
    goals.set_goals([plan])
    asyncio.run(_launch_and_wait(goals, tmp_path))
    assert goals.active_goals == [plan]
