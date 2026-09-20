"""Async policy advice over the newest frame, isolated from the actor."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from ..policy.advice import PolicyGoal, PolicyRequest
from ..policy.candidates import feasible_candidates
from ..policy.state import PolicyState, from_game_state
from ..providers.policy import PolicyAdvisorError
from ..villager_roles import gather_counts, infer_jobs, job_counts

if TYPE_CHECKING:
    from ..providers.policy import PolicyAdvisor
    from .context import LoopContext
    from .snapshot import Perception

log = structlog.stdlib.get_logger()


async def policy_loop(
    ctx: LoopContext,
    advisor: PolicyAdvisor,
    *,
    interval_seconds: float,
) -> None:
    """Refresh cached policy advice without ever blocking the actor."""
    seen_at = 0.0
    try:
        while not ctx.stopping:
            frame = ctx.frames.latest()
            if frame is not None and frame.captured_at > seen_at:
                seen_at = frame.captured_at
                if not frame.alarm:
                    await evaluate_policy_once(ctx, advisor, frame)
            if await _stopped_within(ctx, interval_seconds):
                return
    finally:
        await advisor.aclose()


async def evaluate_policy_once(
    ctx: LoopContext,
    advisor: PolicyAdvisor,
    frame: Perception,
) -> bool:
    """Request and publish one judgment. Expected service failures are safe."""
    request = policy_request(ctx, frame)
    started_at = time.monotonic()
    try:
        advice = await advisor.advise(request)
    except PolicyAdvisorError as exc:
        log.warning(
            "policy_advice_failed",
            frame_tick=frame.tick,
            error=str(exc),
        )
        return False

    ctx.policy_advice.publish(advice)
    log.info(
        "policy_advice_received",
        frame_tick=frame.tick,
        model=advice.model,
        action=advice.action_choice,
        action_confidence=round(advice.action_confidence, 3),
        allocation=advice.allocation_focus,
        allocation_confidence=round(advice.allocation_confidence, 3),
        input_tokens=advice.input_tokens,
        output_tokens=advice.output_tokens,
        latency_ms=round((time.monotonic() - started_at) * 1000),
    )
    return True


def policy_request(ctx: LoopContext, frame: Perception) -> PolicyRequest:
    """Build the narrow, immutable input shared by every policy provider."""
    state = state_for_frame(ctx, frame)
    goals = tuple(
        PolicyGoal(
            name=goal.name,
            metric=goal.metric,
            target=str(goal.target),
            priority=goal.priority,
            progress=goal.progress,
        )
        for goal in ctx.goal_manager.active_goals
    )
    return PolicyRequest(
        source_tick=frame.tick,
        source_captured_at=frame.captured_at,
        state=state,
        candidates=feasible_candidates(state),
        goals=goals,
    )


def state_for_frame(ctx: LoopContext, frame: Perception) -> PolicyState:
    """Create the actor/advisor's common state view from one perception."""
    entities = list(frame.entities)
    jobs = gather_counts(job_counts(infer_jobs(entities))) if entities else {}
    return from_game_state(
        ctx.memory.game_state,
        captured_at=frame.captured_at,
        villager_jobs=jobs,
    )


async def _stopped_within(ctx: LoopContext, interval_seconds: float) -> bool:
    try:
        await asyncio.wait_for(ctx.stop.wait(), timeout=interval_seconds)
    except TimeoutError:
        return False
    return True


__all__ = ["evaluate_policy_once", "policy_loop", "policy_request", "state_for_frame"]
