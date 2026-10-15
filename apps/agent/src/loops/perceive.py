"""The perceive clock: one frame in, one `Perception` out.

The only writer of `memory.game_state` and the build gates, so "how old is this
reading" has one answer. Input-triggered spatial refreshes also read the HUD.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..config import config
from ..entity_snapshot import snapshot_entities
from ..entity_utils import extract_attrs
from ..executor import (
    confirmed_buildings,
    get_detected_entities,
    ledger_policy_state,
    observe_age,
    observe_hud,
    record_observed_buildings,
    set_detected_entities,
    villagers_ordered,
)
from ..goals import THREAT_CLASSES
from ..policy.state import from_game_state
from ..screen import save_screenshot
from ..turn_timing import PERCEIVE_LOOP, TickTimings
from ..window import ensure_game_focused, is_game_running

if TYPE_CHECKING:
    from ..goals import GoalManager
    from ..memory import AgentMemory
    from ..resource_ocr import ResourceReadings
    from .context import LoopContext
    from .snapshot import SpatialRefresh
    from .source import GameSource, RawSpatial, Sighting, StreamCapture

log = structlog.stdlib.get_logger()

# Consecutive focus failures before the run aborts as "lost_focus". Run 1 burned
# 12 of 30 iterations on an unfocusable window with no end-reason label (F-1).
_MAX_FOCUS_FAILURES = 15
_FOCUS_RETRY_DELAY = 1.0
_DETECTION_INTERVAL = 0.2
_HUD_INTERVAL = 0.5
_DETECT_DRAIN_TIMEOUT = 12.0


async def perceive_loop(ctx: LoopContext) -> None:
    """Gate, perceive, pace — until something ends the game. `max_iterations`
    bounds frames: this is the loop that always runs, so it is what to count."""
    from .source import GameSource

    if isinstance(ctx.source, GameSource):
        await _stream_perceive_loop(ctx, ctx.source)
        return
    tick = 0
    while not ctx.stopping:
        if not await _wait_until_playable(ctx):
            return
        request = ctx.frames.pending_spatial_refresh()
        if request is not None:
            await _refresh_spatial_once(ctx, request)
            continue
        tick += 1
        await perceive_hud_once(ctx, tick)
        if ctx.stopping:
            return
        capture_task = asyncio.create_task(perceive_once(ctx, tick))
        refresh_task = asyncio.create_task(ctx.frames.wait_for_spatial_request())
        stop_task = asyncio.create_task(ctx.stop.wait())
        try:
            done, _pending = await asyncio.wait(
                {capture_task, refresh_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done and capture_task not in done:
                capture_task.cancel()
                await asyncio.gather(capture_task, return_exceptions=True)
                return
            if refresh_task in done and capture_task not in done:
                capture_task.cancel()
                await asyncio.gather(capture_task, return_exceptions=True)
                log.info("background_perception_preempted", frame_tick=tick)
                await _refresh_spatial_once(ctx, refresh_task.result())
                tick -= 1  # A canceled frame was never published.
                continue
            await capture_task
        finally:
            refresh_task.cancel()
            stop_task.cancel()
            await asyncio.gather(refresh_task, stop_task, return_exceptions=True)
        if ctx.max_iterations is not None and tick >= ctx.max_iterations:
            ctx.request_stop("iterations_exhausted")
            return
        if ctx.frames.pending_spatial_refresh() is None:
            await ctx.frames.wait_for_due(config.perceive_interval)


async def _stream_perceive_loop(ctx: LoopContext, source: GameSource) -> None:
    """Latest-frame lanes: capture never waits for OCR or detector inference."""
    tick = 0
    next_capture = time.monotonic()
    next_detection = next_capture
    next_hud = next_capture
    capture_task: asyncio.Task[StreamCapture] | None = None
    detect_task: asyncio.Task[RawSpatial | None] | None = None
    hud_task: asyncio.Task[Sighting | None] | None = None
    pending_detection: StreamCapture | None = None
    pending_hud: StreamCapture | None = None
    refresh_task: asyncio.Task[asyncio.Future[SpatialRefresh]] | None = None
    stop_task = asyncio.create_task(ctx.stop.wait())
    try:
        if not await _wait_until_playable(ctx):
            return
        while not ctx.stopping:
            now = time.monotonic()
            if capture_task is None and now >= next_capture:
                capture_task = asyncio.create_task(source.capture_stream())
                next_capture = max(
                    next_capture + config.perceive_interval, now + config.perceive_interval
                )
            if refresh_task is None:
                refresh_task = asyncio.create_task(ctx.frames.wait_for_spatial_request())
            active: set[asyncio.Task[object]] = {stop_task, refresh_task}
            for task in (capture_task, detect_task, hud_task):
                if task is not None:
                    active.add(task)
            done, _ = await asyncio.wait(
                active,
                timeout=(
                    None if capture_task is not None else max(0.0, next_capture - time.monotonic())
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                return
            if refresh_task in done:
                request = refresh_task.result()
                refresh_task = None
                await _refresh_spatial_once(ctx, request)
                continue
            if detect_task is not None and detect_task in done:
                raw = detect_task.result()
                detect_task = None
                if raw is not None:
                    sighting = await source.integrate_spatial(raw)
                    if sighting is not None:
                        _publish_spatial(ctx, sighting.frame.tick, sighting)
            if hud_task is not None and hud_task in done:
                sighting = hud_task.result()
                hud_task = None
                if sighting is not None:
                    _publish_hud(ctx, sighting.frame.tick, sighting)
            if capture_task is not None and capture_task in done:
                capture = capture_task.result()
                capture_task = None
                tick += 1
                if ctx.max_iterations is not None and tick >= ctx.max_iterations:
                    ctx.request_stop("iterations_exhausted")
                if capture.pixels.captured_at >= next_detection:
                    pending_detection = capture
                    next_detection = capture.pixels.captured_at + _DETECTION_INTERVAL
                if capture.pixels.captured_at >= next_hud:
                    pending_hud = capture
                    next_hud = capture.pixels.captured_at + _HUD_INTERVAL
            if ctx.frames.pending_spatial_refresh() is None:
                if detect_task is None and pending_detection is not None:
                    detect_task = asyncio.create_task(
                        source.detect_stream(pending_detection, pending_detection.capture_id)
                    )
                    pending_detection = None
                if hud_task is None and pending_hud is not None:
                    hud_task = asyncio.create_task(
                        source.read_stream_hud(pending_hud, pending_hud.capture_id)
                    )
                    pending_hud = None
    finally:
        tasks: list[asyncio.Task[object]] = []
        for task in (capture_task, hud_task, refresh_task, stop_task):
            if task is not None:
                task.cancel()
                tasks.append(task)
        for task in tasks:
            await asyncio.gather(task, return_exceptions=True)
        # Detector work may be running inside a thread. Cancelling its coroutine
        # would release our capacity while that inference kept running unseen.
        if detect_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(detect_task), timeout=_DETECT_DRAIN_TIMEOUT)
            except TimeoutError:
                log.error("detector_shutdown_drain_timed_out", seconds=_DETECT_DRAIN_TIMEOUT)
                detect_task.cancel()
                await asyncio.gather(detect_task, return_exceptions=True)
            except Exception as exc:
                log.warning("detector_shutdown_failed", error=repr(exc))


async def _refresh_spatial_once(ctx: LoopContext, request: asyncio.Future[SpatialRefresh]) -> None:
    """Service an input-triggered capture before starting another full OCR pass."""
    timings = TickTimings()
    try:
        refresh = await ctx.source.capture_spatial(
            timings, selection_only=ctx.frames.selection_only(request)
        )
        if refresh.spatial_valid and refresh.sighting is not None:
            _publish_spatial(ctx, refresh.sighting.frame.tick, refresh.sighting)
        elif (
            refresh.spatial_valid
            and refresh.entities is not None
            and snapshot_entities(get_detected_entities()) != refresh.entities
        ):
            set_detected_entities(refresh.entities)
        if refresh.spatial_valid and refresh.hud_readings and refresh.sighting is None:
            sync_world_state(
                ctx.memory,
                ctx.goal_manager,
                refresh.hud_readings,
                input_revision=refresh.input_revision,
            )
            if ctx.ledger is not None:
                save_action_evidence(ctx, refresh.screenshot)
        if ctx.ledger is not None:
            ctx.ledger.selected_unit = refresh.selected_unit
            ctx.ledger.selected_at_revision = refresh.input_revision
        ctx.frames.complete_spatial_refresh(request, refresh)
    finally:
        log.info(
            "spatial_refresh_latency",
            total_ms=round(timings.total_ms),
            **{f"{name}_ms": round(value) for name, value in timings.phases.items()},
        )


async def perceive_hud_once(ctx: LoopContext, tick: int) -> None:
    """Publish known economy facts without granting any spatial click authority."""
    timings = TickTimings()
    sighting = await ctx.source.capture_hud(tick, timings)
    if sighting is None:
        return
    _publish_hud(ctx, tick, sighting)


def _publish_hud(ctx: LoopContext, tick: int, sighting: Sighting) -> None:
    frame = sighting.frame
    if ctx.ledger is not None and frame.economic_revision != ctx.ledger.economic_revision:
        return
    if not frame.hud_readings:
        return
    sync_world_state(
        ctx.memory,
        ctx.goal_manager,
        frame.hud_readings,
        input_revision=frame.input_revision,
    )
    ctx.goal_manager.evaluate_progress(ctx.memory.game_state, tick, frozenset(frame.hud_readings))
    ledger = ctx.ledger
    world = (
        ledger_policy_state(ledger, captured_at=frame.captured_at)
        if ledger is not None
        else from_game_state(
            ctx.memory.game_state,
            captured_at=frame.captured_at,
            known_resources=ctx.goal_manager.observed_resource_fields,
        )
    )
    ctx.frames.put_hud(
        replace(
            frame,
            world=replace(world, captured_at=frame.captured_at, spatial_valid=False),
            alarm=ctx.memory.game_state.under_attack,
        )
    )
    log.info("hud_observation_published", frame_tick=tick, age_ms=round(frame.age_ms))


async def _wait_until_playable(ctx: LoopContext) -> bool:
    """Block until the window is focused. False means the run is over.

    Retries are not billed as frames: an unplayable window is not a turn.
    """
    failures = 0
    while not ctx.stopping:
        if not is_game_running():
            log.error("game_not_found", message="AoE2 window not found")
            ctx.request_stop("game_not_found")
            return False
        if ensure_game_focused():
            return True
        failures += 1
        if failures >= _MAX_FOCUS_FAILURES:
            log.error("focus_lost_giving_up", failures=failures)
            ctx.request_stop("lost_focus")
            return False
        log.warning("could_not_focus_game", failures=failures)
        await asyncio.sleep(_FOCUS_RETRY_DELAY)
    return False


async def perceive_once(ctx: LoopContext, tick: int) -> None:
    """One pass: capture, sync the state it feeds, publish the frame."""
    with ctx.latency.tick(PERCEIVE_LOOP, tick) as timings:
        sighting = await ctx.source.capture(tick, timings)
        _publish_spatial(ctx, tick, sighting)


def _publish_spatial(ctx: LoopContext, tick: int, sighting: Sighting) -> None:
    frame = sighting.frame
    ledger = ctx.ledger
    if not frame.spatial_valid or (
        ledger is not None and frame.camera_revision != ledger.camera_revision
    ):
        log.info(
            "perception_discarded",
            frame_tick=frame.tick,
            camera_revision=frame.camera_revision,
            current_camera_revision=ledger.camera_revision if ledger is not None else None,
        )
        ctx.frames.request_now()
        return
    latest_hud = ctx.frames.latest()
    if frame.hud_readings and (
        latest_hud is None
        or latest_hud.hud_captured_at is None
        or (frame.hud_captured_at or frame.captured_at) >= latest_hud.hud_captured_at
    ):
        sync_world_state(
            ctx.memory,
            ctx.goal_manager,
            frame.hud_readings,
            input_revision=frame.input_revision,
        )
        ctx.goal_manager.evaluate_progress(
            ctx.memory.game_state, tick, frozenset(frame.hud_readings)
        )
    if ledger is not None:
        save_action_evidence(ctx, frame.screenshot)
        ledger.spatial_valid = True
    entities = list(frame.entities)
    alarm = ctx.goal_manager.check_alarm(entities, sighting.ownership) if entities else False
    ctx.memory.game_state.under_attack = alarm
    visible_classes = frozenset(extract_attrs(entity).class_name for entity in entities)
    owned_entities = frozenset(
        (attrs.entity_id, attrs.class_name)
        for entity in entities
        if (attrs := extract_attrs(entity)).entity_id in sighting.ownership
        and sighting.ownership[attrs.entity_id][0].value == "own"
    )
    record_observed_buildings(owned_entities)
    ctx.memory.game_state.buildings_seen = confirmed_buildings()
    latest = ctx.frames.latest()
    world = (
        ledger_policy_state(
            ledger,
            captured_at=(latest.hud_captured_at or frame.captured_at)
            if latest is not None
            else frame.captured_at,
        )
        if ledger is not None
        else from_game_state(
            ctx.memory.game_state,
            captured_at=frame.captured_at,
            known_resources=ctx.goal_manager.observed_resource_fields,
        )
    )
    world = replace(
        world,
        idle_count=ctx.memory.game_state.idle_count,
        idle_streak=ctx.memory.game_state.idle_streak,
        visible_classes=visible_classes,
        own_army_present=bool(frozenset(cls for _, cls in owned_entities) & THREAT_CLASSES),
        spatial_valid=True,
    )
    ownership = tuple(
        (entity_id, str(owner), confidence)
        for entity_id, (owner, confidence) in sighting.ownership.items()
    )
    set_detected_entities(frame.entities)
    ctx.frames.put_spatial(replace(frame, alarm=alarm, world=world, ownership=ownership))


def sync_world_state(
    memory: AgentMemory,
    goal_manager: GoalManager,
    hud_readings: ResourceReadings,
    *,
    input_revision: int | None = None,
) -> None:
    """State upkeep from one HUD reading. The perceive loop is its only caller.

    NOT in update_from_observations, which fires more than once per frame: the
    idle streak, buildings-seen and the build gates need one write per reading.
    """
    goal_manager.update_resource_readings(dict(hud_readings), memory)
    game_state = memory.game_state
    game_state.idle_streak = (game_state.idle_streak + 1) if game_state.idle_present else 0
    observe_hud(
        game_state.population,
        game_state.population_cap,
        game_state.resources,
        idle_present=game_state.idle_present,
        idle_count=game_state.idle_count,
        known_resources=goal_manager.observed_resource_fields,
        population_known=goal_manager.observed_population,
        input_revision=input_revision,
        villagers=hud_readings.get("villagers"),
        worker_counts={
            kind: value
            for kind in ("food", "wood", "gold", "stone")
            if isinstance(value := dict(hud_readings).get(f"{kind}_workers"), int)
        },
    )
    observe_age(
        game_state.current_age if "age" in hud_readings else None,
        input_revision=input_revision,
    )
    game_state.buildings_seen = confirmed_buildings()
    game_state.villagers_ordered = villagers_ordered()


def save_action_evidence(ctx: LoopContext, screenshot: bytes) -> None:
    """Keep the observation behind a failure/uncertainty with its operation ID."""
    ledger = ctx.ledger
    if not screenshot or ledger is None:
        return
    for outcome in ledger.outcomes:
        if outcome.status not in {"failed", "uncertain"}:
            continue
        if outcome.operation_id in ledger.outcome_screenshots:
            continue
        path = (
            Path(config.log_dir)
            / "action_screenshots"
            / (f"action_{outcome.operation_id}_{outcome.status}.jpg")
        )
        if ctx.writer is not None:
            if not ctx.writer.submit(path, screenshot, operation_evidence=True):
                continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_screenshot(screenshot, str(path))
        ledger.outcome_screenshots[outcome.operation_id] = str(path)
        log.info(
            "action_evidence_queued" if ctx.writer is not None else "action_evidence_saved",
            action_id=outcome.operation_id,
            status=outcome.status,
            screenshot=str(path),
        )


__all__ = ["perceive_loop", "perceive_once", "save_action_evidence", "sync_world_state"]
