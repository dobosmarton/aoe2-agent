"""The act clock: one decision per frame, executed at once.

Once per frame, because every rule guard reads the HUD: a second decision on
one frame would spend the same state twice. Budget: 100 ms p95 on `decide`.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from ..config import config
from ..models import validate_actions
from ..policy.allocation import focused
from ..policy.candidates import feasible_candidates
from ..policy.engine import decide as fallback_decide
from ..policy.idle import distribute_idle
from ..turn_timing import ACT_LOOP
from .policy import state_for_frame

if TYPE_CHECKING:
    from ..models import Action
    from ..policy.state import PolicyState
    from .context import LoopContext
    from .snapshot import Perception

log = structlog.stdlib.get_logger()

# How long to wait for a frame before checking whether the game ended.
_STOP_POLL = 0.5


async def act_loop(ctx: LoopContext) -> None:
    """Decide on every frame the perceive loop publishes."""
    decided_at = 0.0  # monotonic stamps are positive, so 0.0 is "no frame yet"
    tick = 0
    while not ctx.stopping:
        try:
            frame = await asyncio.wait_for(ctx.frames.after(decided_at), timeout=_STOP_POLL)
        except TimeoutError:
            continue  # no frame yet — re-check the stop flag
        decided_at = frame.captured_at
        tick += 1
        await act_once(ctx, frame, tick)


async def act_once(ctx: LoopContext, frame: Perception, tick: int) -> None:
    """Decide on one frame and execute what it asks for."""
    if ctx.input_lock.locked():
        # The combat tool loop is typing. Queueing behind it would act on a
        # frame the burst has already invalidated.
        log.debug("act_skipped", reason="input_locked", frame_tick=frame.tick)
        return
    with ctx.latency.tick(ACT_LOOP, tick) as timings:
        with timings.phase("decide"):
            actions = _decide(ctx, frame)
        if not actions:
            return
        log.info(
            "act_decided",
            frame_tick=frame.tick,
            state_age_ms=round(frame.age_ms),
            actions=len(actions),
        )
        with timings.phase("execute"):
            # Safe despite the check above: nothing awaits in between, so the lock
            # cannot change hands. Keep `_decide` synchronous or this breaks.
            async with ctx.input_lock:
                results = await ctx.actuator.execute(actions)
        ctx.memory.record_action_results(sum(1 for r in results if r.success), len(results))


def _decide(ctx: LoopContext, frame: Perception) -> list[Action]:
    """Choose from cached advice synchronously, or use the rule fallback."""
    entities = list(frame.entities)
    state = state_for_frame(ctx, frame)
    if frame.alarm:
        return validate_actions(_fallback_commands(ctx, entities, state, frame.alarm))

    candidates = feasible_candidates(state)
    resolution = ctx.policy_advice.consume(
        candidates,
        now=time.monotonic(),
        ttl_seconds=config.policy_advice_ttl,
        minimum_confidence=config.policy_min_confidence,
    )
    if resolution.should_fallback:
        log.debug("policy_advice_rejected", reason=resolution.status)
        return validate_actions(_fallback_commands(ctx, entities, state, frame.alarm))

    allocation = (
        focused(state.age, resolution.allocation_focus)
        if resolution.allocation_focus is not None
        else ctx.goal_manager.allocation
    )
    commands = resolution.action.render() if resolution.action is not None else []
    commands.extend(distribute_idle(entities, state, None, allocation))
    log.info(
        "policy_advice_applied",
        status=resolution.status,
        action=resolution.advice.action_choice if resolution.advice else None,
        allocation=resolution.allocation_focus,
    )
    return validate_actions(commands)


def _fallback_commands(
    ctx: LoopContext,
    entities: list[object],
    state: PolicyState,
    alarm: bool,
) -> list[dict[str, object]]:
    return fallback_decide(
        entities,
        state,
        alarm,
        strategist_allocation=ctx.goal_manager.allocation,
    )


__all__ = ["act_loop", "act_once"]
