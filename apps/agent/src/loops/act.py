"""The act clock: one decision per frame, executed at once.

Once per frame, because every rule guard reads the HUD: a second decision on
one frame would spend the same state twice. Budget: 100 ms p95 on `decide`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from ..models import validate_actions
from ..policy.engine import decide as policy_decide
from ..policy.state import from_game_state
from ..turn_timing import ACT_LOOP
from ..villager_roles import gather_counts, infer_jobs, job_counts

if TYPE_CHECKING:
    from ..models import Action
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
            async with ctx.input_lock:
                results = await ctx.actuator.execute(actions)
        ctx.memory.record_action_results(sum(1 for r in results if r.success), len(results))


def _decide(ctx: LoopContext, frame: Perception) -> list[Action]:
    """The rule engine's answer for this frame. Pure, and microseconds long."""
    entities = list(frame.entities)
    jobs: dict[str, int] = gather_counts(job_counts(infer_jobs(entities))) if entities else {}
    state = from_game_state(
        ctx.memory.game_state,
        captured_at=frame.captured_at,
        villager_jobs=jobs,
    )
    commands = policy_decide(
        entities,
        state,
        frame.alarm,
        strategist_allocation=ctx.goal_manager.allocation,
    )
    return validate_actions(commands)


__all__ = ["act_loop", "act_once"]
