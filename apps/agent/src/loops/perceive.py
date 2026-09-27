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
from ..entity_utils import extract_attrs
from ..executor import (
    confirmed_buildings,
    ledger_policy_state,
    observe_age,
    observe_hud,
    record_observed_buildings,
    villagers_ordered,
)
from ..goals import THREAT_CLASSES
from ..policy.state import from_game_state
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
        sync_world_state(
            ctx.memory,
            ctx.goal_manager,
            frame.hud_readings,
            input_revision=frame.input_revision,
        )
        entities = list(frame.entities)
        alarm = ctx.goal_manager.check_alarm(entities, sighting.ownership) if entities else False
        ledger = ctx.ledger
        if (
            ledger is not None
            and frame.spatial_valid
            and ledger.input_revision == frame.input_revision
        ):
            ledger.spatial_valid = True
        visible_classes = frozenset(extract_attrs(entity).class_name for entity in entities)
        owned_entities = frozenset(
            (attrs.entity_id, attrs.class_name)
            for entity in entities
            if (attrs := extract_attrs(entity)).entity_id in sighting.ownership
            and sighting.ownership[attrs.entity_id][0].value == "own"
        )
        building_evidence = frozenset(
            (attrs.entity_id, attrs.class_name)
            for entity in entities
            if (attrs := extract_attrs(entity)).entity_id not in sighting.ownership
            or sighting.ownership[attrs.entity_id][0].value != "enemy"
        )
        if frame.spatial_valid and (
            ledger is None or ledger.input_revision == frame.input_revision
        ):
            record_observed_buildings(building_evidence)
        ctx.memory.game_state.buildings_seen = confirmed_buildings()
        owned_classes = frozenset(cls for _, cls in owned_entities)
        own_army_present = bool(owned_classes & THREAT_CLASSES)
        world = (
            ledger_policy_state(ledger)
            if ledger is not None
            else from_game_state(
                ctx.memory.game_state,
                captured_at=frame.captured_at,
                known_resources=ctx.goal_manager.observed_resource_fields,
            )
        )
        world = replace(
            world,
            idle_count=ctx.memory.game_state.idle_count,
            idle_streak=ctx.memory.game_state.idle_streak,
            captured_at=frame.captured_at,
            visible_classes=visible_classes,
            own_army_present=own_army_present,
            spatial_valid=frame.spatial_valid,
        )
        ownership = tuple(
            (entity_id, str(owner), confidence)
            for entity_id, (owner, confidence) in sighting.ownership.items()
        )
        ctx.frames.put(replace(frame, alarm=alarm, world=world, ownership=ownership))


def sync_world_state(
    memory: AgentMemory,
    goal_manager: GoalManager,
    hud_readings: ResourceReadings,
    *,
    input_revision: int | None = None,
) -> None:
    """State upkeep from one HUD reading. The perceive loop is its only caller.

    NOT in update_from_observations, which fires more than once per frame: the
    idle streak, buildings-seen and the build gates need one write per reading.
    """
    goal_manager.update_resource_readings(dict(hud_readings), memory)
    game_state = memory.game_state
    game_state.idle_streak = (game_state.idle_streak + 1) if game_state.idle_present else 0
    observe_hud(
        game_state.population,
        game_state.population_cap,
        game_state.resources,
        idle_present=game_state.idle_present,
        known_resources=goal_manager.observed_resource_fields,
        population_known=goal_manager.observed_population,
        input_revision=input_revision,
    )
    observe_age(
        game_state.current_age if "age" in hud_readings else None,
        input_revision=input_revision,
    )
    game_state.buildings_seen = confirmed_buildings()
    game_state.villagers_ordered = villagers_ordered()


__all__ = ["perceive_loop", "perceive_once", "sync_world_state"]
