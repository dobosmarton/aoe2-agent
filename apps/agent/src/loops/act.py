"""The actor: ask once, revalidate against current facts, execute one named action."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from ..config import config
from ..models import validate_actions
from ..policy.allocation import focused
from ..policy.candidates import feasible_candidates, find_candidate
from ..policy.fallback import select_fallback
from ..policy.request import policy_request, state_for_frame
from ..providers.policy import PolicyAdvisorError
from ..turn_timing import ACT_LOOP

if TYPE_CHECKING:
    from ..policy.advice import PolicyAdvice, PolicyRequest
    from ..policy.candidates import ActionCandidate
    from ..providers.policy import PolicyAdvisor
    from .context import LoopContext
    from .snapshot import Perception

log = structlog.stdlib.get_logger()
_STOP_POLL = 0.5


async def act_loop(ctx: LoopContext, advisor: PolicyAdvisor) -> None:
    """Consume the execution frame, not merely the frame that started the ask."""
    consumed_at = 0.0
    tick = 0
    try:
        while not ctx.stopping:
            try:
                frame = await asyncio.wait_for(ctx.frames.after(consumed_at), timeout=_STOP_POLL)
            except TimeoutError:
                continue
            tick += 1
            consumed_at = max(consumed_at, await act_once(ctx, advisor, frame, tick))
    finally:
        await advisor.aclose()


async def act_once(
    ctx: LoopContext,
    advisor: PolicyAdvisor,
    frame: Perception,
    tick: int,
) -> float:
    """One bounded judgment, with the latest observation checked under input ownership."""
    if frame.alarm or ctx.input_lock.locked():
        log.debug(
            "act_skipped", reason="alarm" if frame.alarm else "input_locked", frame_tick=frame.tick
        )
        return frame.captured_at

    request = policy_request(ctx, frame)
    started_at = time.monotonic()
    advice: PolicyAdvice | None = None
    failure: str | None = None
    with ctx.latency.tick(ACT_LOOP, tick) as timings:
        with timings.phase("policy"):
            try:
                async with asyncio.timeout(config.policy_timeout):
                    advice = await advisor.advise(request)
            except TimeoutError:
                failure = "provider_timeout"
            except PolicyAdvisorError as exc:
                failure = f"provider_error:{type(exc).__name__}"
            except Exception as exc:
                failure = f"provider_error:{type(exc).__name__}"
                log.warning("policy_provider_failed", error=repr(exc))

        if ctx.input_lock.locked():
            latest = ctx.frames.latest() or frame
            _log_outcome(frame, latest, started_at, "superseded", "input_locked")
            return latest.captured_at

        with timings.phase("execute"):
            async with ctx.input_lock:
                latest = ctx.frames.latest() or frame
                if latest.alarm:
                    _log_outcome(frame, latest, started_at, "superseded", "alarm")
                    return latest.captured_at
                if not latest.spatial_valid or (
                    ctx.ledger is not None and latest.input_revision != ctx.ledger.input_revision
                ):
                    _log_outcome(
                        frame, latest, started_at, "superseded", "observation_crossed_input"
                    )
                    return latest.captured_at

                current_state = state_for_frame(ctx, latest)
                candidates = feasible_candidates(current_state)
                selected, status, reason = _choose_action(
                    ctx, request, advice, failure, latest, candidates
                )
                _log_outcome(
                    frame,
                    latest,
                    started_at,
                    status,
                    reason,
                    action=selected.id,
                    model=advice.model if advice is not None else None,
                )
                if selected.id == "tactical_handoff":
                    ctx.tactical_requested.set()
                    return latest.captured_at
                commands = selected.render()
                if not commands:
                    return latest.captured_at
                actions = validate_actions(commands)
                if len(actions) != len(commands):
                    log.error("catalog_render_invalid", action=selected.id)
                    if ctx.ledger is not None:
                        ctx.ledger.record_failure(f"invalid catalog render: {selected.id}")
                    return latest.captured_at

                log.info(
                    "act_decided",
                    action=selected.id,
                    source_tick=frame.tick,
                    execution_tick=latest.tick,
                    source_input_revision=request.source_input_revision,
                    state_age_ms=round(latest.age_ms),
                    reservations=ctx.ledger.reservations() if ctx.ledger is not None else {},
                )
                failures_before = ctx.ledger.failure_count if ctx.ledger is not None else 0
                operation_before = ctx.ledger.next_operation_id if ctx.ledger is not None else 0
                results = await ctx.actuator.execute(actions)
                ctx.memory.record_action_results(
                    sum(result.success for result in results), len(results)
                )
                if ctx.ledger is not None:
                    if (
                        (not results or not all(result.success for result in results))
                        and ctx.ledger.failure_count == failures_before
                        and not ctx.ledger.has_pending_since(operation_before)
                    ):
                        detail = next(
                            (result.detail for result in results if not result.success), "no result"
                        )
                        ctx.ledger.record_failure(f"{selected.id}: {detail}")
                    elif selected.id.startswith(("assign_",)):
                        ctx.ledger.record_success()
                log.info(
                    "act_execution_outcome",
                    action=selected.id,
                    source_tick=frame.tick,
                    execution_tick=latest.tick,
                    success=bool(results) and all(result.success for result in results),
                    details=[result.detail for result in results],
                )
                return latest.captured_at


def _choose_action(
    ctx: LoopContext,
    request: PolicyRequest,
    advice: PolicyAdvice | None,
    failure: str | None,
    latest: Perception,
    candidates: tuple[ActionCandidate, ...],
) -> tuple[ActionCandidate, str, str]:
    """A newer frame is fine; a changed fact or input needs a fresh eligible choice."""
    state = state_for_frame(ctx, latest)
    if failure is not None:
        status = "timed_out" if failure == "provider_timeout" else "fallback"
        return select_fallback(candidates, state, ctx.goal_manager.allocation), status, failure
    if advice is None:
        return (
            select_fallback(candidates, state, ctx.goal_manager.allocation),
            "fallback",
            "no_advice",
        )

    reason: str | None = None
    if (
        advice.source_tick != request.source_tick
        or advice.source_captured_at != request.source_captured_at
    ):
        reason = "source_frame_mismatch"
    elif request.source_input_revision != latest.input_revision:
        reason = "input_revision_changed"
    elif request.goal_revision != ctx.goal_manager.revision:
        reason = "goal_revision_changed"
    elif request.state.age != state.age:
        reason = "age_changed"
    elif advice.action_confidence < config.policy_min_confidence:
        reason = "low_confidence"
    else:
        selected = find_candidate(candidates, advice.action_choice)
        if selected is not None:
            return selected, "applied", "eligible_on_latest_frame"
        reason = "candidate_unavailable"

    allocation = ctx.goal_manager.allocation
    if advice.allocation_confidence >= config.policy_min_confidence:
        allocation = focused(state.age, advice.allocation_focus)
    return select_fallback(candidates, state, allocation), "rejected", reason


def _log_outcome(
    source: Perception,
    execution: Perception,
    started_at: float,
    status: str,
    reason: str,
    **details: object,
) -> None:
    log.info(
        "policy_advice_outcome",
        status=status,
        reason=reason,
        source_tick=source.tick,
        execution_tick=execution.tick,
        revalidated=execution.tick != source.tick,
        state_age_ms=round(execution.age_ms),
        latency_ms=round((time.monotonic() - started_at) * 1000),
        **details,
    )


__all__ = ["act_loop", "act_once"]
