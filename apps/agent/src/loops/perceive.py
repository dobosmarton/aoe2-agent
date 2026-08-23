"""The perceive clock: one frame in, one `Perception` out.

The only writer of `memory.game_state` and the build gates, so "how old is this
reading" has one answer. Budget: 2 s p95. It waits on nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

import structlog

from ..config import config
from ..executor import confirmed_buildings, observe_age, observe_hud, villagers_ordered
from ..turn_timing import PERCEIVE_LOOP
from ..window import ensure_game_focused, is_game_running

if TYPE_CHECKING:
    from ..goals import GoalManager
    from ..memory import AgentMemory
    from ..resource_ocr import ResourceReadings
    from .context import LoopContext

log = structlog.stdlib.get_logger()

# Consecutive focus failures before the run aborts as "lost_focus". Run 1 burned
# 12 of 30 iterations on an unfocusable window with no end-reason label (F-1).
_MAX_FOCUS_FAILURES = 15
_FOCUS_RETRY_DELAY = 1.0


async def perceive_loop(ctx: LoopContext) -> None:
    """Gate, perceive, pace — until something ends the game. `max_iterations`
    bounds frames: this is the loop that always runs, so it is what to count."""
    tick = 0
    while not ctx.stopping:
        if not await _wait_until_playable(ctx):
            return
        tick += 1
        await perceive_once(ctx, tick)
        if ctx.max_iterations is not None and tick >= ctx.max_iterations:
            ctx.request_stop("iterations_exhausted")
            return
        await ctx.frames.wait_for_due(config.perceive_interval)


async def _wait_until_playable(ctx: LoopContext) -> bool:
    """Block until the window is focused. False means the run is over.

    Retries are not billed as frames: an unplayable window is not a turn.
    """
    failures = 0
    while not ctx.stopping:
        if not is_game_running():
            log.error("game_not_found", message="AoE2 window not found")
            ctx.request_stop("game_not_found")
            return False
        if ensure_game_focused():
            return True
        failures += 1
        if failures >= _MAX_FOCUS_FAILURES:
            log.error("focus_lost_giving_up", failures=failures)
            ctx.request_stop("lost_focus")
            return False
        log.warning("could_not_focus_game", failures=failures)
        await asyncio.sleep(_FOCUS_RETRY_DELAY)
    return False


async def perceive_once(ctx: LoopContext, tick: int) -> None:
    """One pass: capture, sync the state it feeds, publish the frame."""
    with ctx.latency.tick(PERCEIVE_LOOP, tick) as timings:
        sighting = await ctx.source.capture(tick, timings)
        frame = sighting.frame
        sync_world_state(ctx.memory, ctx.goal_manager, frame.hud_readings)
        entities = list(frame.entities)
        alarm = ctx.goal_manager.check_alarm(entities, sighting.ownership) if entities else False
        ctx.frames.put(replace(frame, alarm=alarm))


def sync_world_state(
    memory: AgentMemory,
    goal_manager: GoalManager,
    hud_readings: ResourceReadings,
) -> None:
    """State upkeep from one HUD reading. The perceive loop is its only caller.

    NOT in update_from_observations, which fires more than once per frame: the
    idle streak, buildings-seen and the build gates need one write per reading.
    """
    if hud_readings:
        goal_manager.update_resource_readings(dict(hud_readings), memory)
    game_state = memory.game_state
    game_state.idle_streak = (game_state.idle_streak + 1) if game_state.idle_present else 0
    game_state.buildings_seen = confirmed_buildings()
    game_state.villagers_ordered = villagers_ordered()
    observe_hud(
        game_state.population,
        game_state.population_cap,
        game_state.resources,
        idle_present=game_state.idle_present,
    )
    observe_age(game_state.current_age)


__all__ = ["perceive_loop", "perceive_once", "sync_world_state"]
