"""The deliberate clock: the LLM, on exceptions only.

It writes parameters, and its actions are discarded on a routine tick. Only
combat and a pop cap let it press keys, and then it holds the input lock.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

import structlog

from ..config import config
from ..executor import is_pop_capped
from ..memory import STUCK_LOOP_THRESHOLD
from ..policy.allocation import is_famine
from ..policy.state import from_game_state
from ..strategist_phase import _maybe_launch_strategist
from ..turn_phases import (
    _build_llm_context,
    _execute_turn_actions,
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

# The exception list from IMPROVEMENT-PLAN.md P3.1.
Trigger = Literal["alarm", "housed", "stuck", "goals", "famine", "interval"]
# The 2 that may act. Both need mid-turn rescans and composite tools that the
# single-shot Action union cannot express.
_ACTING: frozenset[Trigger] = frozenset({"alarm", "housed"})
# How long to wait for a frame before checking whether the game ended.
_STOP_POLL = 0.5


async def deliberate_loop(
    ctx: LoopContext,
    strategist: StrategistProvider,
    provider: ExecutorProvider,
) -> None:
    """Ask the LLM only when something exceptional happened."""
    seen, tick = 0.0, 0
    strategist_task: asyncio.Task[None] | None = None
    while not ctx.stopping:
        try:
            frame = await asyncio.wait_for(ctx.frames.after(seen), timeout=_STOP_POLL)
        except TimeoutError:
            continue
        seen = frame.captured_at
        tick += 1
        strategist_task = _maybe_launch_strategist(
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
        trigger = _trigger(ctx, frame, tick)
        if trigger is not None:
            await deliberate_once(ctx, provider, frame, tick, trigger)


def _trigger(ctx: LoopContext, frame: Perception, tick: int) -> Trigger | None:
    """Why the LLM should run this frame, or None to stay quiet."""
    if frame.alarm:
        return "alarm"
    if is_pop_capped():
        return "housed"
    if ctx.memory.no_change_streak() >= STUCK_LOOP_THRESHOLD:
        return "stuck"
    if ctx.goal_manager.take_goals_changed():
        return "goals"
    # The whole state, for one field: `is_famine` owns the threshold, so this
    # keeps it in one place. It ignores `captured_at`.
    state = from_game_state(ctx.memory.game_state, captured_at=frame.captured_at)
    if is_famine(state):
        return "famine"
    if config.deliberate_interval > 0 and tick % config.deliberate_interval == 0:
        return "interval"
    return None


async def deliberate_once(
    ctx: LoopContext,
    provider: ExecutorProvider,
    frame: Perception,
    tick: int,
    trigger: Trigger,
) -> None:
    """One LLM turn: act on an exception, else plan and discard."""
    with ctx.latency.tick(DELIBERATE_LOOP, tick) as timings:
        with timings.phase("context"):
            context = _build_llm_context(
                ctx.memory,
                ctx.goal_manager,
                frame.entity_summary,
                list(frame.entities),
            )
        with timings.phase("executor"):
            turn = _act_turn if trigger in _ACTING else _plan_turn
            response = await turn(ctx, provider, context, frame, tick)
    reason = check_game_over(response, ctx.memory, tick)
    if reason:
        ctx.request_stop(reason)


async def _act_turn(
    ctx: LoopContext,
    provider: ExecutorProvider,
    context: str,
    frame: Perception,
    tick: int,
) -> LLMResult:
    """Combat and housing: the tool loop presses its own keys, so hold the lock."""
    async with ctx.input_lock:
        response = await provider.act(context, frame.width, frame.height)
        actions = record_llm_turn(response, ctx.memory, ctx.goal_manager, tick, ctx.goal_logger)
        await _execute_or_record(response, actions, ctx.memory, tick)
    return response


async def _plan_turn(
    ctx: LoopContext,
    provider: ExecutorProvider,
    context: str,
    frame: Perception,
    tick: int,
) -> LLMResult:
    """Routine: the rules own the acting, so the LLM's clicks are dropped."""
    response = await provider.plan(context, frame.width, frame.height)
    actions = record_llm_turn(response, ctx.memory, ctx.goal_manager, tick, ctx.goal_logger)
    if actions:
        # What the rules overrode. Phase 6.3 reads this to size the gap.
        log.info("llm_actions_discarded", count=len(actions), tick=tick)
    return response


async def _execute_or_record(
    response: LLMResult,
    actions: list[dict[str, object]],
    memory: AgentMemory,
    tick: int,
) -> None:
    """Execute the turn's actions, or only record them when the tool loop
    already ran them during its own call."""
    if response.get("actions_already_executed"):
        success = response.get("success_count", len(actions))
        memory.record_action_results(success, len(actions))
        log.info("actions_executed", iteration=tick, total=len(actions), successful=success)
        return
    await _execute_turn_actions(actions, tick, memory, response.get("reasoning", ""))


__all__ = ["deliberate_loop", "deliberate_once"]
