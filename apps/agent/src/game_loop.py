"""The supervisor: it starts the clocks and labels the end.

The clocks live in `loops/`. This module owns only what a game needs once.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    import numpy as np

    from .game_profile import GameProfile
    from .overlay import DetectionOverlay
    from .providers.base import ChatWire
    from .providers.executor_provider import ExecutorProvider
    from .providers.policy import PolicyAdvisor

from .config import config
from .detection_phase import (
    DetectionUnavailableError,
    DetectorFactory,
    init_detector,
)
from .diagnostic_writer import DiagnosticWriter
from .executor import (
    ActionLedger,
    bind_ledger,
    clear_refresh_callbacks,
    set_absolute_zoom,
    set_command_button,
    set_modifier_map,
    set_placement_refresh_fn,
    set_preview_capture_fn,
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
from .screen import capture_pixels_after
from .turn_phases import get_ground_commands
from .window import ensure_game_focused, game_platform

log = structlog.stdlib.get_logger()

_PERMISSION_FIX = (
    "run the agent from Terminal.app and allow Terminal in System Settings > "
    "Privacy & Security > Screen Recording and Accessibility"
)


def _missing_macos_permissions() -> list[str]:
    from .window_macos import input_allowed, screen_recording_allowed

    checks = (("screen_recording", screen_recording_allowed()), ("accessibility", input_allowed()))
    return [name for name, allowed in checks if not allowed]


def _untypeable_agent_keys() -> list[str]:
    """Send keys through the active Mac layout; the agent keys it can't type.

    Runs on the main thread, before any key press, as macOS requires.
    """
    from .keyboard_macos import use_layout_keys

    layout, missing = use_layout_keys()
    if missing:
        log.error(
            "keyboard_layout_cannot_type",
            layout=layout.name,
            keys=missing,
            fix="add a layout that types these keys without Option, and select it",
        )
    return missing


_POPUP_SAMPLES = 3
_POPUP_SAMPLE_GAP = 0.4


def _popup_open() -> bool:
    """Whether a menu or popup dims the game. Input would land on it instead.

    Only frames captured after this call count, and every one must be dimmed: an
    open menu stays, while the first frame after the game comes forward can be
    left over from before (a run from Terminal refused on one such frame).
    """
    from .game_screen import hud_brightness, popup_open
    from .screen import capture_native

    asked = time.monotonic()
    readings: list[tuple[float, float]] = []
    while len(readings) < _POPUP_SAMPLES:
        pixels = capture_native()
        if pixels.captured_at < asked:
            time.sleep(0.05)
            continue
        rgb = pixels.pixels()
        if not popup_open(rgb):
            return False
        readings.append(hud_brightness(rgb))
        time.sleep(_POPUP_SAMPLE_GAP)
    log.info("game_dimmed", hud_brightness=[(round(a), round(b)) for a, b in readings])
    return True


async def _prepare_macos_game(profile: GameProfile | None) -> str | None:
    """Bring the game forward once, then fix its zoom and modifier keys. Returns why
    the game isn't ready, or None.

    This is the only place the agent pulls the game to the front. After this, input
    stops while another app is in front, so switching away stops the agent.
    """
    if not ensure_game_focused():
        log.error("game_not_in_front", message="the game window couldn't be brought forward")
        return "lost_focus"
    if await asyncio.to_thread(_popup_open):
        log.error("game_menu_open", fix="close the game's menu or popup, then start the agent")
        return "menu_open"
    set_modifier_map(profile.modifier_map.items() if profile is not None else ())
    zoom_steps = profile.zoom_steps if profile is not None else 0
    zoom = await set_absolute_zoom(zoom_steps)
    if not zoom.success:
        log.warning("zoom_not_set", detail=zoom.detail, zoom_steps=zoom_steps)
    return None


@runtime_checkable
class _AsyncClose(Protocol):
    async def aclose(self) -> None: ...


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
    profile: GameProfile | None,
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
    writer = DiagnosticWriter()
    return LoopContext(
        memory=memory,
        goal_manager=goal_manager,
        goal_logger=GoalLogger(log_dir),
        source=GameSource(
            detector=(detector_factory or init_detector)() if use_detection else None,
            overlay=overlay,
            screenshots_dir=screenshots_dir,
            ledger=ledger,
            profile=profile,
            writer=writer,
        ),
        actuator=GameActuator(),
        ledger=ledger,
        writer=writer,
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
        if game_platform() == "macos" and (missing := _missing_macos_permissions()):
            log.error("macos_permission_missing", missing=missing, fix=_PERMISSION_FIX)
            _request_stop(ctx, memory, "missing_permission")
            return memory
        if game_platform() == "macos" and _untypeable_agent_keys():
            _request_stop(ctx, memory, "keyboard_layout")
            return memory
        from .game_profile import load_profile, platform_problem

        profile = load_profile(config.game_profile_path) if config.game_profile_path else None
        if (problem := platform_problem(profile, game_platform())) is not None:
            log.error("game_profile_unusable", problem=problem)
            _request_stop(ctx, memory, "game_profile")
            return memory
        _use_profile(profile)
        ctx = _build_context(
            memory,
            use_detection=use_detection,
            use_overlay=use_overlay,
            time_budget=time_budget,
            max_iterations=max_iterations,
            detector_factory=detector_factory,
            ledger=ledger,
            profile=profile,
        )
        memory.latency = ctx.latency
        # Perception owns captures; selection/HUD reads need not wait for YOLO.
        refresh = frame_refresh(ctx.frames)
        set_rescan_fn(refresh)
        set_rescan_full_fn(refresh)
        set_selection_refresh_fn(frame_refresh(ctx.frames, selection_only=True))
        set_placement_refresh_fn(frame_refresh(ctx.frames, skip_selection=True))
        set_preview_capture_fn(_capture_after)

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
        relative_zoom = True
        if game_platform() == "macos":
            if (not_ready := await _prepare_macos_game(profile)) is not None:
                _request_stop(ctx, memory, not_ready)
                return memory
            relative_zoom = False
        # Before the first frame on purpose: the scout explores during the
        # slowest perception pass of the game (engine warm-up).
        await ctx.actuator.execute(
            validate_actions(get_ground_commands(1, relative_zoom=relative_zoom))
        )
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
    if isinstance(ctx.source, _AsyncClose):
        await ctx.source.aclose()
    else:
        ctx.source.close()
    # Exits that bypass both except clauses (CancelledError is a BaseException)
    # reach here unlabelled — run 13 logged an empty game_end_reason on a manual
    # stop (T-543). This is the one choke point.
    if not memory.game_end_reason:
        memory.game_end_reason = "interrupted"
    if ctx.ledger is not None:
        ctx.ledger.finalize_unverified()
        save_action_evidence(ctx, ctx.frames.evidence_screenshot())
    if ctx.writer is not None:
        await ctx.writer.close()
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


async def _capture_after(after: float) -> np.ndarray:
    """A frame drawn after `after`, captured off the event loop for the placement preview."""
    return await asyncio.to_thread(capture_pixels_after, after)


def _use_profile(profile: GameProfile | None) -> None:
    """Order units with the profile's mouse button, and log what the run uses."""
    command_button = profile.command_button if profile is not None else "right"
    set_command_button(command_button)
    log.info(
        "game_profile_loaded",
        path=str(config.game_profile_path) if config.game_profile_path else None,
        platform=profile.platform if profile is not None else None,
        command_button=command_button,
        zoom_steps=profile.zoom_steps if profile is not None else 0,
    )


def _request_stop(ctx: LoopContext | None, memory: AgentMemory, reason: str) -> None:
    if ctx is None:
        memory.game_end_reason = reason
        return
    ctx.request_stop(reason)
