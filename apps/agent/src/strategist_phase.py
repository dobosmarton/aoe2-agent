"""Strategist scheduling: launch the goal-updating LLM call as a background task.

The strategist is a separate (smaller, cheaper) LLM call that decides which
high-level goals the executor should pursue. We run it concurrently with the
executor so the executor never waits on it. Two responsibilities:

  - `_run_strategist_async`: the actual call body — generate goals, push them
    into `goal_manager`, log progress.
  - `maybe_launch_strategist`: gating logic — runs only when the strategist
    cadence/alarm logic says it should and never doubles up an in-flight call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .goal_logger import GoalLogger
    from .goals import GoalManager
    from .memory import AgentMemory
    from .providers.strategist import StrategistProvider
    from .resource_ocr import ResourceReadings

log = structlog.stdlib.get_logger()


@dataclass(frozen=True, slots=True)
class StrategistView:
    """What one frame gives the strategist to plan from."""

    alarm: bool
    entity_summary: str
    hud_readings: ResourceReadings
    known_buildings: str
    brief: str = ""


async def _run_strategist_async(
    strategist: StrategistProvider,
    iteration: int,
    view: StrategistView,
    memory: AgentMemory,
    goal_manager: GoalManager,
    goal_logger: GoalLogger,
) -> None:
    """Invoke the strategist to create/update goals (runs as background task)."""
    try:
        prev_goals = list(goal_manager.active_goals)
        # The game loop's per-turn HUD reading is passed in — the strategist
        # never re-OCRs the frame. Its returned readings are the same object,
        # so nothing is written back to game_state from here.
        new_goals, _readings = await strategist.generate_goals(
            memory.game_state,
            goal_manager.get_goals_summary(),
            view.entity_summary,
            iteration,
            alarm=view.alarm,
            readings=view.hud_readings,
            known_buildings=view.known_buildings,
            brief=view.brief,
        )
        goal_manager.set_goals(new_goals)
        goal_manager.set_allocation(strategist.last_allocation)
        goal_logger.log_goals_created(iteration, new_goals)
        if prev_goals:
            goal_logger.log_strategist_update(iteration, prev_goals, new_goals)
        log.info(
            "strategist_goals_updated", turn=iteration, goal_count=len(new_goals), alarm=view.alarm
        )
    except Exception as e:
        log.warning("strategist_failed", error=str(e))


def maybe_launch_strategist(
    strategist: StrategistProvider,
    iteration: int,
    view: StrategistView,
    memory: AgentMemory,
    goal_manager: GoalManager,
    goal_logger: GoalLogger,
    pending_task: asyncio.Task | None,
) -> asyncio.Task | None:
    """Launch the strategist as a background task if its goals are due.

    Returns the new task, or the existing pending task if one is still running.
    """
    if not strategist.should_run(
        alarm=view.alarm,
        age=memory.game_state.current_age,
        goals_left=bool(goal_manager.active_goals),
    ):
        return pending_task

    if pending_task is not None and not pending_task.done():
        log.debug("strategist_skipped", reason="previous_task_pending")
        return pending_task

    task = asyncio.create_task(
        _run_strategist_async(strategist, iteration, view, memory, goal_manager, goal_logger),
        name=f"strategist_turn_{iteration}",
    )
    log.info("strategist_launched_async", turn=iteration, alarm=view.alarm)
    return task


__all__ = ["StrategistView", "maybe_launch_strategist"]
