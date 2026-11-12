"""The actor: ask once, revalidate against current facts, execute one named action."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING

import structlog

from ..config import config
from ..executor import ActionOutcome
from ..models import validate_actions
from ..policy.allocation import focused
from ..policy.candidates import feasible_candidates, find_candidate
from ..policy.fallback import select_fallback
from ..policy.request import policy_request, state_for_frame
from ..providers.policy import PolicyAdvisorError
from ..turn_timing import ACT_LOOP
from .perceive import save_action_evidence

if TYPE_CHECKING:
    from ..policy.advice import PolicyAdvice, PolicyRequest
    from ..policy.candidates import ActionCandidate
    from ..policy.controller import AgentController
    from ..policy.state import PolicyState
    from ..providers.policy import PolicyAdvisor
    from .context import LoopContext
    from .snapshot import Perception

log = structlog.stdlib.get_logger()
_STOP_POLL = 0.5
_POLICY_START_INTERVAL = 0.5
_POLICY_PREEMPT_POLL = 0.1


async def act_loop(ctx: LoopContext, advisor: PolicyAdvisor) -> None:
    """Consume the execution frame, not merely the frame that started the ask."""
    consumed_version = 0
    last_started_at = -_POLICY_START_INTERVAL
    tick = 0
    try:
        while not ctx.stopping:
            try:
                frame = await asyncio.wait_for(
                    ctx.frames.after_version(consumed_version), timeout=_STOP_POLL
                )
            except TimeoutError:
                continue
            remaining = last_started_at + _POLICY_START_INTERVAL - time.monotonic()
            if remaining > 0 and not frame.alarm:
                await asyncio.sleep(remaining)
                frame = ctx.frames.latest() or frame
            if ctx.stopping:
                return
            last_started_at = time.monotonic()
            tick += 1
            consumed_version = max(consumed_version, await act_once(ctx, advisor, frame, tick))
    finally:
        await advisor.aclose()


async def act_once(
    ctx: LoopContext,
    advisor: PolicyAdvisor,
    frame: Perception,
    tick: int,
) -> int:
    """One bounded judgment, with the latest observation checked under input ownership."""
    if ctx.stopping:
        return frame.publication_version
    if frame.alarm or ctx.input_lock.locked():
        log.debug(
            "act_skipped", reason="alarm" if frame.alarm else "input_locked", frame_tick=frame.tick
        )
        return frame.publication_version

    request = policy_request(ctx, frame)
    started_at = time.monotonic()
    advice: PolicyAdvice | None = None
    failure: str | None = None
    if _forced_obligation(ctx, request.state, request.candidates) is not None:
        failure = "obligation_overdue"
    with ctx.latency.tick(ACT_LOOP, tick) as timings:
        with timings.phase("policy"):
            if failure is None:
                advice, failure = await _await_advice(ctx, advisor, request, frame)

        if failure == "shutdown":
            latest = ctx.frames.latest() or frame
            _log_outcome(frame, latest, started_at, "superseded", "shutdown")
            return latest.publication_version

        if ctx.input_lock.locked():
            latest = ctx.frames.latest() or frame
            _log_outcome(frame, latest, started_at, "superseded", "input_locked")
            return latest.publication_version

        with timings.phase("execute"):
            async with ctx.input_lock:
                latest = ctx.frames.latest() or frame
                if latest.alarm:
                    _log_outcome(frame, latest, started_at, "superseded", "alarm")
                    return latest.publication_version
                if (not latest.spatial_valid and not latest.hud_only) or (
                    ctx.ledger is not None and latest.input_revision != ctx.ledger.input_revision
                ):
                    _log_outcome(
                        frame, latest, started_at, "superseded", "observation_crossed_input"
                    )
                    return latest.publication_version

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
                    **_advice_details(advice),
                )
                if selected.id == "tactical_handoff":
                    ctx.tactical_requested.set()
                    return latest.publication_version
                commands = selected.render()
                if not commands:
                    return latest.publication_version
                actions = validate_actions(commands)
                if len(actions) != len(commands):
                    log.error("catalog_render_invalid", action=selected.id)
                    if ctx.ledger is not None:
                        ctx.ledger.record_outcome(
                            ActionOutcome(
                                ctx.ledger.new_operation_id(),
                                selected.id,
                                "failed",
                                "invalid catalog render",
                            )
                        )
                        save_action_evidence(ctx, ctx.frames.evidence_screenshot())
                    return latest.publication_version

                log.info(
                    "act_decided",
                    action=selected.id,
                    source_tick=frame.tick,
                    execution_tick=latest.tick,
                    source_input_revision=request.source_input_revision,
                    state_age_ms=round(latest.age_ms),
                    reservations=ctx.ledger.reservations() if ctx.ledger is not None else {},
                )
                operation_before = ctx.ledger.next_operation_id if ctx.ledger is not None else 0
                results = await ctx.actuator.execute(actions)
                ctx.controller.record_attempt(selected.id)
                ctx.memory.record_action_results(
                    sum(result.success for result in results), len(results)
                )
                if (
                    ctx.ledger is not None
                    and (not results or not all(result.success for result in results))
                    and not any(
                        outcome.operation_id >= operation_before for outcome in ctx.ledger.outcomes
                    )
                ):
                    detail = next(
                        (result.detail for result in results if not result.success), "no result"
                    )
                    ctx.ledger.record_outcome(
                        ActionOutcome(ctx.ledger.new_operation_id(), selected.id, "failed", detail)
                    )
                if ctx.ledger is not None:
                    save_action_evidence(ctx, ctx.frames.evidence_screenshot())
                log.info(
                    "act_execution_outcome",
                    action=selected.id,
                    source_tick=frame.tick,
                    execution_tick=latest.tick,
                    success=bool(results) and all(result.success for result in results),
                    outcome_pending=(
                        ctx.ledger is not None and ctx.ledger.has_pending_since(operation_before)
                    ),
                    details=[result.detail for result in results],
                )
                return latest.publication_version


async def _await_advice(
    ctx: LoopContext,
    advisor: PolicyAdvisor,
    request: PolicyRequest,
    source: Perception,
) -> tuple[PolicyAdvice | None, str | None]:
    """Await one model call while observations may make it unsafe or overdue."""
    advice_task = asyncio.create_task(advisor.advise(request))
    stop_task = asyncio.create_task(ctx.stop.wait())
    seen_version = source.publication_version
    deadline = time.monotonic() + config.policy_timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "provider_timeout"
            frame_task = asyncio.create_task(ctx.frames.after_version(seen_version))
            try:
                done, _ = await asyncio.wait(
                    {advice_task, stop_task, frame_task},
                    timeout=min(remaining, _POLICY_PREEMPT_POLL),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    return None, "shutdown"
                latest = ctx.frames.latest() or source
                if latest.alarm:
                    return None, "alarm"
                if ctx.ledger is not None and latest.input_revision != ctx.ledger.input_revision:
                    return None, "intervening_execution"
                if latest.input_revision != request.source_input_revision:
                    return None, "intervening_execution"
                state = state_for_frame(ctx, latest)
                if _forced_obligation(ctx, state, feasible_candidates(state)) is not None:
                    return None, "obligation_overdue"
                if advice_task in done:
                    try:
                        return advice_task.result(), None
                    except PolicyAdvisorError as exc:
                        return None, f"provider_error:{type(exc).__name__}"
                    except Exception as exc:
                        log.warning("policy_provider_failed", error=repr(exc))
                        return None, f"provider_error:{type(exc).__name__}"
                if frame_task in done:
                    seen_version = frame_task.result().publication_version
            finally:
                frame_task.cancel()
                await asyncio.gather(frame_task, return_exceptions=True)
    finally:
        advice_task.cancel()
        stop_task.cancel()
        await asyncio.gather(advice_task, stop_task, return_exceptions=True)


def _forced_obligation(
    ctx: LoopContext, state: PolicyState, candidates: tuple[ActionCandidate, ...]
) -> ActionCandidate | None:
    """An overdue obligation that runs without waiting for Jev, since it can't decline it."""
    overdue = ctx.controller.overdue(state, candidates, ctx.goal_manager.allocation)
    if overdue is None or ctx.controller.advisor_may_decline(overdue.id):
        return None
    return overdue


@dataclass(frozen=True, slots=True)
class _Judgment:
    """Jev's answer, checked against the latest frame."""

    reason: str
    choice: ActionCandidate | None = None
    confident: bool = False

    @property
    def confident_choice(self) -> ActionCandidate | None:
        return self.choice if self.confident else None


def _choose_action(
    ctx: LoopContext,
    request: PolicyRequest,
    advice: PolicyAdvice | None,
    failure: str | None,
    latest: Perception,
    candidates: tuple[ActionCandidate, ...],
) -> tuple[ActionCandidate, str, str]:
    """Jev decides; the obligations and the fallback list only fill its gaps.

    An overdue obligation runs unless Jev confidently chose otherwise and the
    obligation lets it (`advisor_may_decline`). Live, keeping the Town Center
    busy queued villagers at 538 and 584 food while the goals said to bank 500
    for the Feudal Age.
    """
    state = state_for_frame(ctx, latest)
    stale = _stale_reason(request, advice, latest, state, ctx.goal_manager.revision)
    usable = advice is not None and failure is None and stale is None
    if usable and advice.allocation_confidence >= config.policy_min_confidence:
        ctx.goal_manager.set_allocation(focused(state.age, advice.allocation_focus))
    judgment = _judge(advice, failure or stale, candidates)
    confident = judgment.confident_choice
    overdue = ctx.controller.overdue(state, candidates, ctx.goal_manager.allocation)
    if overdue is not None and not _declines(ctx.controller, overdue, confident, request):
        return overdue, "fallback", "obligation_overdue"
    if confident is not None:
        reason = "declined_obligation" if overdue is not None else judgment.reason
        return confident, "applied", reason
    fallback = select_fallback(candidates, state, ctx.goal_manager.allocation, request.goals)
    if fallback.id == "wait" and judgment.choice is not None and judgment.choice.id != "wait":
        # The list had nothing to do; Jev's best guess beats doing nothing.
        return judgment.choice, "applied", "fallback_had_nothing"
    return fallback, _fallback_status(failure, advice), judgment.reason


def _declines(
    controller: AgentController,
    obligation: ActionCandidate,
    choice: ActionCandidate | None,
    request: PolicyRequest,
) -> bool:
    """Whether Jev confidently chose something else with the obligation on offer.

    A "wait" when the obligation wasn't among Jev's options says nothing about
    it: in game 7 every confident "wait" before a forced villager queue came
    from a request that didn't offer a villager.
    """
    if choice is None or choice.id == obligation.id:
        return False
    offered = any(candidate.id == obligation.id for candidate in request.candidates)
    return offered and controller.advisor_may_decline(obligation.id)


def _stale_reason(
    request: PolicyRequest,
    advice: PolicyAdvice | None,
    latest: Perception,
    state: PolicyState,
    goal_revision: int,
) -> str | None:
    """Why Jev's answer no longer fits the latest frame, or None if it still does."""
    if advice is None:
        return "no_advice"
    if (
        advice.source_tick != request.source_tick
        or advice.source_captured_at != request.source_captured_at
    ):
        return "source_frame_mismatch"
    if request.source_input_revision != latest.input_revision:
        return "input_revision_changed"
    if request.goal_revision != goal_revision:
        return "goal_revision_changed"
    if request.state.age != state.age:
        return "age_changed"
    return None


def _judge(
    advice: PolicyAdvice | None,
    unusable_reason: str | None,
    candidates: tuple[ActionCandidate, ...],
) -> _Judgment:
    """Jev's pick on the latest frame, and whether it was sure enough to act on."""
    if advice is None or unusable_reason is not None:
        return _Judgment(unusable_reason or "no_advice")
    choice = find_candidate(candidates, advice.action_choice)
    if choice is None:
        return _Judgment("candidate_unavailable")
    if advice.action_confidence < config.policy_min_confidence:
        return _Judgment("low_confidence", choice)
    return _Judgment("eligible_on_latest_frame", choice, confident=True)


def _fallback_status(failure: str | None, advice: PolicyAdvice | None) -> str:
    if failure == "provider_timeout":
        return "timed_out"
    return "fallback" if failure is not None or advice is None else "rejected"


_RANKED_IN_LOG = 3


def _advice_details(advice: PolicyAdvice | None) -> dict[str, object]:
    """Jev's own answer, so a run review can see what it ranked, not just what ran."""
    if advice is None:
        return {}
    ranked = sorted(advice.action_probabilities.items(), key=itemgetter(1), reverse=True)
    return {
        "advised": advice.action_choice,
        "advised_confidence": round(advice.action_confidence, 2),
        "advised_ranking": {action: round(p, 2) for action, p in ranked[:_RANKED_IN_LOG]},
        "allocation_focus": advice.allocation_focus,
    }


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
