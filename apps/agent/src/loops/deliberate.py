"""Deliberate work only for combat, requested handoff, and bounded recovery."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Literal

import structlog

from ..strategist_phase import maybe_launch_strategist
from ..turn_phases import (
    build_llm_context,
    check_game_over,
    known_buildings_line,
    record_llm_turn,
)
from ..turn_timing import DELIBERATE_LOOP

if TYPE_CHECKING:
    from ..memory import AgentMemory
    from ..providers.base import LLMResult
    from ..providers.executor_provider import ExecutorProvider
    from ..providers.strategist import StrategistProvider
    from .context import LoopContext
    from .snapshot import Perception

log = structlog.stdlib.get_logger()
Trigger = Literal["alarm", "handoff", "recovery"]
_STOP_POLL = 0.5
_RECOVERY_COOLDOWN = 30.0
_FAMINE_STALL = 30.0


async def deliberate_loop(
    ctx: LoopContext,
    strategist: StrategistProvider,
    provider: ExecutorProvider,
) -> None:
    """Own the background strategist and any in-flight recovery on shutdown."""
    seen, tick = 0.0, 0
    strategist_task: asyncio.Task[None] | None = None
    last_recovery = 0.0
    last_food: int | None = None
    food_progress_at = time.monotonic()
    try:
        while not ctx.stopping:
            try:
                frame = await asyncio.wait_for(ctx.frames.after(seen), timeout=_STOP_POLL)
            except TimeoutError:
                continue
            seen = frame.captured_at
            tick += 1
            strategist_task = maybe_launch_strategist(
                strategist,
                tick,
                frame.alarm,
                ctx.memory,
                ctx.goal_manager,
                frame.entity_summary,
                frame.hud_readings,
                known_buildings_line(list(frame.entities)),
                ctx.goal_logger,
                strategist_task,
            )
            state = frame.world
            if state is not None and "food" in state.known_resources:
                if last_food is None or state.food > last_food:
                    food_progress_at = time.monotonic()
                last_food = state.food
            trigger = _trigger(ctx, frame, food_progress_at, last_recovery)
            if trigger is not None:
                if trigger == "handoff":
                    ctx.tactical_requested.clear()
                if trigger == "recovery":
                    last_recovery = time.monotonic()
                await deliberate_once(ctx, provider, frame, tick, trigger)
    finally:
        if strategist_task is not None:
            strategist_task.cancel()
            await asyncio.gather(strategist_task, return_exceptions=True)


def _trigger(
    ctx: LoopContext,
    frame: Perception,
    food_progress_at: float,
    last_recovery: float,
) -> Trigger | None:
    if frame.alarm:
        return "alarm"
    if ctx.tactical_requested.is_set():
        return "handoff"
    if time.monotonic() - last_recovery < _RECOVERY_COOLDOWN:
        return None
    failed = ctx.ledger is not None and ctx.ledger.failure_streak >= 3
    state = frame.world
    famine_stalled = (
        state is not None
        and "food" in state.known_resources
        and state.food < 60
        and time.monotonic() - food_progress_at >= _FAMINE_STALL
    )
    return "recovery" if failed or famine_stalled else None


async def deliberate_once(
    ctx: LoopContext,
    provider: ExecutorProvider,
    frame: Perception,
    tick: int,
    trigger: Trigger,
) -> None:
    """Act within one owned input window; never discard a routine model plan."""
    with ctx.latency.tick(DELIBERATE_LOOP, tick) as timings:
        with timings.phase("context"):
            context = build_llm_context(
                ctx.memory,
                ctx.goal_manager,
                frame.entity_summary,
                list(frame.entities),
                observed_state=frame.world,
            )
            context = f"Trigger: {trigger}. Recent failures: {ctx.ledger.recent_failures if ctx.ledger else []}.\n{context}"
        with timings.phase("executor"):
            async with ctx.input_lock:
                if trigger == "recovery":
                    response = await provider.act_recovery(context, frame.width, frame.height)
                else:
                    response = await provider.act(context, frame.width, frame.height)
                actions = record_llm_turn(
                    response, ctx.memory, ctx.goal_manager, tick, ctx.goal_logger
                )
                await _execute_or_record(response, actions, ctx.memory, tick)
    reason = check_game_over(response, ctx.memory, tick)
    if reason:
        ctx.request_stop(reason)


async def _execute_or_record(
    response: LLMResult,
    actions: list[dict[str, object]],
    memory: AgentMemory,
    tick: int,
) -> None:
    if response.get("actions_already_executed"):
        success = response.get("success_count", len(actions))
        memory.record_action_results(success, len(actions))
        log.info("actions_executed", iteration=tick, total=len(actions), successful=success)
        return
    # The deliberate provider may only act through guarded tools. A plain
    # returned action list is reasoning output, never an alternate input path.
    log.warning("unexecuted_deliberate_actions_rejected", iteration=tick, count=len(actions))


__all__ = ["deliberate_loop", "deliberate_once"]
