"""Deliberate work only for combat, requested handoff, and bounded recovery."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

from ..providers.executor_provider import ToolExecutionGate
from ..strategist_phase import maybe_launch_strategist
from ..turn_phases import (
    build_llm_context,
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
_MAX_RECOVERY_CYCLES = 3


@dataclass(slots=True)
class RecoveryBudget:
    """Stop retrying an essential failure when evidence never improves."""

    cycles: int = 0
    last_progress: tuple[float | None, float | None] | None = None

    def observe(self, food_at: float | None, tc_at: float | None) -> None:
        progress = (food_at, tc_at)
        if self.last_progress is not None and progress != self.last_progress:
            self.cycles = 0
        self.last_progress = progress

    def use_cycle(self) -> bool:
        if self.cycles >= _MAX_RECOVERY_CYCLES:
            return False
        self.cycles += 1
        return True


async def deliberate_loop(
    ctx: LoopContext,
    strategist: StrategistProvider,
    provider: ExecutorProvider,
) -> None:
    """Own the background strategist and any in-flight recovery on shutdown."""
    seen, tick = 0.0, 0
    strategist_task: asyncio.Task[None] | None = None
    last_recovery = 0.0
    recovery_budget = RecoveryBudget()
    try:
        while not ctx.stopping:
            try:
                frame = await asyncio.wait_for(ctx.frames.after(seen), timeout=_STOP_POLL)
            except TimeoutError:
                continue
            seen = frame.captured_at
            tick += 1
            if ctx.ledger is not None:
                recovery_budget.observe(ctx.ledger.food_progress_at, ctx.ledger.tc_progress_at)
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
            trigger = None if frame.hud_only else _trigger(ctx, frame, last_recovery)
            if trigger is not None:
                if trigger == "handoff":
                    ctx.tactical_requested.clear()
                if trigger == "recovery":
                    if not recovery_budget.use_cycle():
                        log.error(
                            "essential_progress_unverified",
                            recovery_cycles=recovery_budget.cycles,
                            recent_failures=ctx.ledger.recent_failures if ctx.ledger else [],
                        )
                        ctx.request_stop("essential_progress_unverified")
                        return
                    last_recovery = time.monotonic()
                await deliberate_once(ctx, provider, frame, tick, trigger)
    finally:
        if strategist_task is not None:
            strategist_task.cancel()
            await asyncio.gather(strategist_task, return_exceptions=True)


def _trigger(
    ctx: LoopContext,
    frame: Perception,
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
    stalled = state is not None and (
        (state.tc_stalled and state.pending_villagers == 0)
        or (state.food_stalled and not state.assignment_pending)
    )
    if failed or stalled:
        log.warning(
            "recovery_triggered",
            failed_attempts=ctx.ledger.failure_streak if ctx.ledger else 0,
            tc_stalled=state.tc_stalled if state else False,
            food_stalled=state.food_stalled if state else False,
        )
        return "recovery"
    return None


async def deliberate_once(
    ctx: LoopContext,
    provider: ExecutorProvider,
    frame: Perception,
    tick: int,
    trigger: Trigger,
) -> None:
    """Let the model reason freely; each returned tool validates under input ownership."""
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
            gate = ToolExecutionGate(
                input_lock=ctx.input_lock,
                current_revision=lambda: (
                    ctx.ledger.input_revision if ctx.ledger is not None else frame.input_revision
                ),
                source_revision=frame.input_revision,
            )
            if trigger == "recovery":
                response = await provider.act_recovery(
                    context, frame.width, frame.height, gate=gate
                )
            else:
                response = await provider.act(context, frame.width, frame.height, gate=gate)
            actions = record_llm_turn(response, ctx.memory, ctx.goal_manager, tick, ctx.goal_logger)
            await _execute_or_record(response, actions, ctx.memory, tick)
    # Text returned by a model is not screen evidence of victory or defeat.
    # A visual end-game reader must own that transition when qualified.


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
