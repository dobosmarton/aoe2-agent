"""The supervisor: it starts the clocks and labels the end.

The clocks live in `loops/`. This module owns only what a game needs once.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .overlay import DetectionOverlay
    from .providers.base import ChatWire
    from .providers.executor_provider import ExecutorProvider
    from .providers.policy import PolicyAdvisor

from .config import config
from .detection_phase import (
    DETECTION_AVAILABLE,
    ENTITY_DISPLAY_LIMIT,
    DetectionUnavailableError,
    DetectorFactory,
    init_detector,
    init_frame_differ,
)
from .entity_utils import build_entity_summary
from .executor import (
    ActionLedger,
    bind_ledger,
    clear_detected_entities,
    clear_refresh_callbacks,
    execute_actions,
    set_detected_entities,
    set_rescan_fn,
    set_rescan_full_fn,
    set_selection_refresh_fn,
    unbind_ledger,
)
from .goal_logger import GoalLogger
from .goals import GoalManager
from .loops.act import act_loop
from .loops.context import LoopContext
from .loops.deliberate import deliberate_loop
from .loops.perceive import perceive_loop, save_action_evidence
from .loops.source import GameActuator, GameSource, frame_refresh
from .memory import AgentMemory
from .models import validate_actions
from .providers.strategist import StrategistProvider, get_default_goals
from .resource_ocr import warm_up_ocr
from .screen import capture_screenshot, save_screenshot
from .turn_phases import get_ground_commands

log = structlog.stdlib.get_logger()


@runtime_checkable
class _SerializableEntity(Protocol):
    def to_dict(self) -> object: ...


def _serialize_entity(entity: object) -> object:
    """Cross the optional detector serialization boundary without using Any."""
    return entity.to_dict() if isinstance(entity, _SerializableEntity) else entity


def _warm_up_wire(wire: ChatWire) -> None:
    """Import the wire's 115 deferred modules now. Doing it inside the first
    call once blocked the loop for 2 minutes (run 2026-08-20). Never raises."""
    try:
        started = time.monotonic()
        wire.warm_up()
        log.info("wire_warmed", seconds=round(time.monotonic() - started, 1))
    except Exception as e:
        log.warning("wire_warmup_failed", error=str(e))


def _start_warm_ups(provider: ExecutorProvider) -> list[asyncio.Task[None]]:
    """Pay the slow imports off the loop. The OCR engine alone costs 10-15 s on
    the VM and was the bulk of the post-startup freeze (F-6)."""
    tasks = [asyncio.create_task(asyncio.to_thread(_warm_up_wire, provider.wire))]
    if config.ocr_backend == "rapidocr":
        tasks.append(asyncio.create_task(asyncio.to_thread(warm_up_ocr)))
    return tasks


def _open_overlay() -> DetectionOverlay | None:
    """The debug overlay, or None. It is never worth failing a game for."""
    try:
        # Aliased because the name is already bound under TYPE_CHECKING; the
        # import is deferred because tkinter is absent on some hosts.
        from .overlay import DetectionOverlay as Overlay

        overlay = Overlay()
    except Exception as e:
        log.warning("overlay_init_failed", error=str(e))
        return None
    log.info("overlay_enabled")
    return overlay


def _build_context(
    memory: AgentMemory,
    *,
    use_detection: bool,
    use_overlay: bool,
    time_budget: float | None,
    max_iterations: int | None,
    detector_factory: DetectorFactory | None,
    ledger: ActionLedger,
) -> LoopContext:
    """Everything the clocks share, wired to the real game."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = None
    if config.save_screenshots:
        screenshots_dir = log_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)

    overlay = _open_overlay() if use_overlay else None
    goal_manager = GoalManager()
    goal_manager.set_goals(get_default_goals(turn=0))
    from .game_profile import load_profile

    profile = load_profile(config.game_profile_path) if config.game_profile_path else None
    return LoopContext(
        memory=memory,
        goal_manager=goal_manager,
        goal_logger=GoalLogger(log_dir),
        source=GameSource(
            detector=(detector_factory or init_detector)() if use_detection else None,
            overlay=overlay,
            frame_differ=init_frame_differ(),
            screenshots_dir=screenshots_dir,
            ledger=ledger,
            profile=profile,
        ),
        actuator=GameActuator(),
        ledger=ledger,
        time_budget=time_budget,
        max_iterations=max_iterations,
    )


async def _stop_on_budget(ctx: LoopContext) -> None:
    """End the game when the time budget runs out. Idle without one."""
    if ctx.time_budget is None:
        await ctx.stop.wait()
        return
    remaining = ctx.time_budget - ctx.memory.get_game_duration_seconds()
    try:
        await asyncio.wait_for(ctx.stop.wait(), timeout=max(remaining, 0.0))
    except TimeoutError:
        log.info("time_budget_reached", seconds=ctx.time_budget)
        ctx.request_stop("timeout")


async def _run_clocks(
    ctx: LoopContext,
    strategist: StrategistProvider,
    provider: ExecutorProvider,
    policy_advisor: PolicyAdvisor,
) -> None:
    """Run the clocks until one of them ends the game, then stop the rest."""
    tasks = [
        asyncio.create_task(perceive_loop(ctx), name="perceive"),
        asyncio.create_task(act_loop(ctx, policy_advisor), name="act"),
        asyncio.create_task(deliberate_loop(ctx, strategist, provider), name="deliberate"),
        asyncio.create_task(_stop_on_budget(ctx), name="budget"),
    ]
    try:
        # FIRST_COMPLETED, not gather: a clock only returns when the game is
        # over, and waiting for the rest would hold shutdown for the length of
        # an in-flight LLM call.
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task  # re-raise whatever ended it
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        close = getattr(strategist, "aclose", None)
        if close is not None:
            await close()


async def game_loop(
    provider: ExecutorProvider,
    policy_advisor: PolicyAdvisor,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,
    use_overlay: bool = False,
    detector_factory: DetectorFactory | None = None,
) -> AgentMemory:
    """Play one game on the perceive, act-with-policy and deliberate clocks.

    `max_iterations` bounds PERCEIVE FRAMES, about 0.5 s each, where a turn used
    to be 10. Use `time_budget` to bound a real game."""
    if memory is None:
        memory = AgentMemory()
    memory.start_game()
    ledger = ActionLedger()
    memory.action_ledger = ledger
    ledger_token = bind_ledger(ledger)

    warm_ups = _start_warm_ups(provider)
    ctx: LoopContext | None = None
    clocks_started = False
    try:
        ctx = _build_context(
            memory,
            use_detection=use_detection,
            use_overlay=use_overlay,
            time_budget=time_budget,
            max_iterations=max_iterations,
            detector_factory=detector_factory,
            ledger=ledger,
        )
        memory.latency = ctx.latency
        # Perception owns captures; selection/HUD reads need not wait for YOLO.
        refresh = frame_refresh(ctx.frames)
        set_rescan_fn(refresh)
        set_rescan_full_fn(refresh)
        set_selection_refresh_fn(frame_refresh(ctx.frames, selection_only=True))

        memory.memories_loaded = list(provider.loaded_memory_titles)
        log.info(
            "game_loop_start",
            loop_arch="clocks",
            detection=use_detection,
            iteration_budget=max_iterations,
            iteration_counts="frames",
            time_budget=time_budget,
            memories=len(memory.memories_loaded),
        )
        # Before the first frame on purpose: the scout explores during the
        # slowest perception pass of the game (engine warm-up).
        await ctx.actuator.execute(validate_actions(get_ground_commands(1)))
        clocks_started = True
        await _run_clocks(ctx, StrategistProvider(), provider, policy_advisor)
    except KeyboardInterrupt:
        log.info("game_loop_interrupted")
        _request_stop(ctx, memory, "interrupted")
    except DetectionUnavailableError as exc:
        log.error(
            "detector_unavailable",
            error_type=type(exc).__name__,
            error=repr(exc),
        )
        _request_stop(ctx, memory, "detector_unavailable")
        raise
    except Exception as e:
        log.error("game_loop_error", error_type=type(e).__name__, error=repr(e))
        _request_stop(ctx, memory, "error")
        raise
    finally:
        try:
            if not clocks_started:
                await policy_advisor.aclose()
            if ctx is None:
                await _stop_warm_ups(warm_ups)
                log.info("game_metrics_final", **memory.get_metrics_snapshot())
            else:
                await _close_out(ctx, memory, warm_ups)
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
        finally:
            clear_refresh_callbacks()
            unbind_ledger(ledger_token)

    return memory


async def _close_out(
    ctx: LoopContext,
    memory: AgentMemory,
    warm_ups: list[asyncio.Task[None]],
) -> None:
    """Stop the warm-ups, release the source, and label the end."""
    await _stop_warm_ups(warm_ups)
    ctx.source.close()
    # Exits that bypass both except clauses (CancelledError is a BaseException)
    # reach here unlabelled — run 13 logged an empty game_end_reason on a manual
    # stop (T-543). This is the one choke point.
    if not memory.game_end_reason:
        memory.game_end_reason = "interrupted"
    if ctx.ledger is not None:
        ctx.ledger.finalize_unverified()
        save_action_evidence(ctx.ledger, ctx.frames.evidence_screenshot())
    log.info("game_metrics_final", **memory.get_metrics_snapshot())
    ctx.goal_logger.log_game_end(
        memory.turn_count,
        memory.game_end_reason,
        len(ctx.goal_manager.completed_goals),
    )


async def _stop_warm_ups(warm_ups: list[asyncio.Task[None]]) -> None:
    for warm_up in warm_ups:
        warm_up.cancel()
    await asyncio.gather(*warm_ups, return_exceptions=True)


def _request_stop(ctx: LoopContext | None, memory: AgentMemory, reason: str) -> None:
    if ctx is None:
        memory.game_end_reason = reason
        return
    ctx.request_stop(reason)


# ---------------------------------------------------------------------------
# Single iteration (testing / debugging)
# ---------------------------------------------------------------------------


async def run_single_iteration(
    provider: ExecutorProvider,
    memory: AgentMemory | None = None,
    execute: bool = False,
    use_detection: bool = True,
) -> dict:
    """Run a single iteration of the game loop for testing."""
    if memory is None:
        memory = AgentMemory()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    screenshot, width, height = capture_screenshot()

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = log_dir / f"test_{timestamp}.jpg"
    save_screenshot(screenshot, str(screenshot_path))

    detected_entities: Sequence[object] = []
    if use_detection and DETECTION_AVAILABLE:
        try:
            from detection.inference.detector import get_detector

            detector = get_detector(use_mock=False, model_name=config.detection_model)
            detected_entities = detector.detect(screenshot)
            set_detected_entities(detected_entities)
        except Exception as e:
            log.warning("detection_failed", error=str(e))

    context = memory.get_context_for_llm()
    if detected_entities:
        summary = build_entity_summary(list(detected_entities), max_count=ENTITY_DISPLAY_LIMIT)
        entity_context = (
            "\n## Detected Entities (from YOLO)\n"
            "Use target_class or target_id to interact with these:\n" + summary + "\n"
        )
        context = entity_context + "\n" + context

    response = await provider.get_actions(context, width, height)

    memory.create_turn(
        reasoning=response.get("reasoning", ""),
        actions=response.get("actions", []),
        observations=response.get("observations", {}),
    )

    actions = response.get("actions") or []
    if execute and actions:
        await execute_actions(actions)

    clear_detected_entities()

    return {
        "screenshot_path": str(screenshot_path),
        "reasoning": response.get("reasoning", ""),
        "observations": response.get("observations", {}),
        "actions": response.get("actions", []),
        "memory_context": context,
        "detected_entities": [_serialize_entity(entity) for entity in detected_entities],
    }
