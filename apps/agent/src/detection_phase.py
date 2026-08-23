"""Vision-pipeline glue between the game loop and the YOLO detector.

Owns:
  - `_init_detector` / `_init_frame_differ`: optional-resource initialization
    (the agent can run without a detector).
  - `_register_rescan_callbacks`: hooks the executor's rescan-on-keypress paths
    into the detector + overlay. Lives here because the closures `_rescan` and
    `_rescan_full` capture `detector`/`overlay`/`frame_differ` together.
  - `_capture_screenshot` / `_run_detection` / `_classify_entities`: the per-turn
    screenshot → detection → ownership-tagging chain.

The `Detector` alias unifies `EntityDetector` (local YOLO) and `RemoteDetector`
(HTTP server). They share a duck-typed surface — `.tracker`, `.use_mock`,
`.backend`, `.confidence_threshold`, plus the methods invoked through
`_invoke_detector(...)` — but don't share a base class.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from detection.inference.detector import DetectedEntity, EntityDetector
    from detection.inference.frame_diff import FrameChange, FrameDiffer
    from detection.inference.remote_detector import RemoteDetector

    from .overlay import DetectionOverlay

    Detector = EntityDetector | RemoteDetector

from .config import config
from .entity_utils import CLASSES_BY_KIND, build_entity_summary
from .executor import (
    GATE_BUILDING_CLASSES,
    clear_detected_entities,
    get_detected_entities,
    set_detected_entities,
    set_rescan_fn,
    set_rescan_full_fn,
)
from .screen import capture_screenshot, save_screenshot
from .window import get_game_window_rect

log = structlog.stdlib.get_logger()


try:
    from detection.inference.detector import get_detector

    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    log.info("detection_not_available", message="Running without YOLO detection")


ENTITY_DISPLAY_LIMIT = 20
RESCAN_SCREENSHOT_QUALITY = 50
TRACKER_CONFIDENCE_THRESHOLD = 0.8
ENTITY_DROP_RATIO = 0.5
FRAME_DIFFER_THRESHOLD = 0.03


async def _invoke_detector(
    det: Detector, method: str, *args: object, **kwargs: object
) -> list[DetectedEntity]:
    """Call a detector method, handling both sync and async implementations.

    All EntityDetector / RemoteDetector inference methods return
    `list[DetectedEntity]` — pyright can't see that through `getattr`,
    so the return type is asserted here.
    """
    fn = cast("Callable[..., object]", getattr(det, method))
    if asyncio.iscoroutinefunction(fn):
        return cast("list[DetectedEntity]", await fn(*args, **kwargs))
    return cast("list[DetectedEntity]", await asyncio.to_thread(fn, *args, **kwargs))


def _init_detector() -> Detector | None:
    """Initialize YOLO detector (remote or local)."""
    if not DETECTION_AVAILABLE:
        return None
    try:
        if config.detection_host:
            from detection.inference.remote_detector import get_remote_detector

            detector = get_remote_detector(
                config.detection_host,
                imgsz=config.detection_imgsz,
                model_name=config.detection_model,
            )
            log.info("detector_initialized", mode="remote", server=config.detection_host)
            return detector
        # adaptive_sahi gates SAHI everywhere: it picks detect_adaptive vs the
        # single-pass detect() below, and use_sahi keeps detect() from tiling.
        detector = get_detector(
            use_mock=False,
            imgsz=config.detection_imgsz,
            use_sahi=config.adaptive_sahi,
            model_name=config.detection_model,
        )
        backend = "mock" if detector.use_mock else detector.backend or "yolo"
        log.info(
            "detector_initialized", mode=backend, confidence_threshold=detector.confidence_threshold
        )
        return detector
    except Exception as e:
        log.warning("detector_init_failed", error=str(e))
        return None


# Classes that never move, so a cached position stays valid once translated by
# the camera pan. Animals are excluded by name: CLASSES_BY_KIND["food"] mixes
# berry bushes with sheep, boar and deer.
_HERD_CLASSES: frozenset[str] = frozenset({"sheep", "boar", "deer"})
STATIC_CLASSES: frozenset[str] = (
    frozenset().union(*CLASSES_BY_KIND.values()) | GATE_BUILDING_CLASSES
) - _HERD_CLASSES

# Phase-correlation confidence below which a pan is not trusted. Run
# 2026_08_22_1 measured a bimodal split (p50 0.82, p10 0.25); anything from 0.5
# to 0.7 divides the same 36 of 59 turn-to-turn pairs.
PAN_CONFIDENCE_MIN = 0.7


def _translated_static(entities: list[dict], shift: tuple[float, float]) -> list[dict]:
    """The static entities, moved by the camera pan.

    `shift` is how far the CONTENT moved, so it adds. Mobile classes are dropped
    rather than translated: a villager that walked is worse than one the caller
    knows it has not been told about.
    """
    dx, dy = shift
    moved: list[dict] = []
    for entity in entities:
        if entity.get("class") not in STATIC_CLASSES:
            continue
        cx, cy = entity.get("center", (0, 0))
        shifted = dict(entity)
        shifted["center"] = (int(cx + dx), int(cy + dy))
        moved.append(shifted)
    return moved


# Why a pan could not serve a rescan. "" means it did.
PanRefusal = Literal["", "low_confidence", "empty_cache", "nothing_static_cached"]


def _pan_translation(change: FrameChange, cached: list[dict]) -> tuple[list[dict], PanRefusal]:
    """The cached static map moved to the new view, or why it cannot be.

    One function so the caller can log WHY a pan was refused: run 2026_08_22_2
    translated 0 of 195 rescans and the log never said which clause declined.
    """
    if change.response < PAN_CONFIDENCE_MIN:
        return [], "low_confidence"
    if not cached:
        return [], "empty_cache"
    translated = _translated_static(cached, change.shift)
    if not translated:
        return [], "nothing_static_cached"
    return translated, ""


def _init_frame_differ() -> FrameDiffer | None:
    """Initialize frame differ for skipping redundant rescans."""
    try:
        from detection.inference.frame_diff import FrameDiffer

        return FrameDiffer(threshold=FRAME_DIFFER_THRESHOLD)
    except ImportError:
        return None


def _register_rescan_callbacks(
    detector: Detector,
    overlay: DetectionOverlay | None,
    frame_differ: FrameDiffer | None,
) -> None:
    """Register rescan + full detection callbacks on the executor module."""

    async def _rescan() -> None:
        if overlay:
            overlay.hide()
        screenshot, _, _ = capture_screenshot(quality=RESCAN_SCREENSHOT_QUALITY)

        change = frame_differ.compare(screenshot) if frame_differ else None
        if change and not change.changed:
            if (
                detector.tracker
                and detector.tracker.get_confidence() > TRACKER_CONFIDENCE_THRESHOLD
            ):
                predicted = detector.tracker.predict()
                set_detected_entities(predicted)
                if overlay:
                    overlay.show(predicted, get_game_window_rect())
                    log.debug("rescan_predicted", entity_count=len(predicted))
                return
            log.debug("rescan_skipped", reason="no_change")
            if overlay:
                overlay.show(detector._previous_entities, get_game_window_rect())
                return

        # The view only panned, so the static map is still known — translate it
        # rather than pay for a detection (run 2026_08_22_1: 112 of 212).
        # A disabled cache stays silent so an A/B run's log carries only signal.
        if config.rescan_cache and change:
            translated, declined = _pan_translation(change, get_detected_entities())
            if translated:
                set_detected_entities(translated)
                log.debug(
                    "rescan_translated",
                    entity_count=len(translated),
                    shift=[round(v) for v in change.shift],
                    response=round(change.response, 3),
                )
                if overlay:
                    overlay.show(translated, get_game_window_rect())
                return
            log.debug("rescan_pan_declined", reason=declined, response=round(change.response, 3))

        entities = await _invoke_detector(detector, "detect_fast_multi", screenshot)
        if (
            detector.tracker
            and detector._previous_entities
            and len(entities) < len(detector._previous_entities) * ENTITY_DROP_RATIO
        ):
            detector.tracker.reset()
            log.debug("tracker_reset", reason="camera_moved")
        set_detected_entities(entities)
        if overlay:
            overlay.show(entities, get_game_window_rect())
            log.debug("rescan_complete", entity_count=len(entities), mode="fast")

    async def _rescan_full() -> None:
        if overlay:
            overlay.hide()
        screenshot_full, _, _ = capture_screenshot(quality=85)
        if frame_differ:
            frame_differ.reset()
        full_method = "detect" if config.adaptive_sahi else "detect_fast"
        entities = await _invoke_detector(detector, full_method, screenshot_full)
        if detector.tracker:
            detector.tracker.reset()
        set_detected_entities(entities)
        if overlay:
            overlay.show(entities, get_game_window_rect())
        log.info("rescan_full_complete", entity_count=len(entities))

    set_rescan_fn(_rescan)
    set_rescan_full_fn(_rescan_full)


async def _capture_screenshot(
    overlay: DetectionOverlay | None,
    screenshots_dir: Path | None,
    iteration: int,
) -> tuple[bytes, int, int]:
    """Capture game screenshot, optionally saving to disk."""
    if overlay:
        overlay.hide()
    screenshot, width, height = capture_screenshot()
    log.debug("screenshot_captured", width=width, height=height)

    if config.save_screenshots and screenshots_dir:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = screenshots_dir / f"{timestamp}_{iteration:05d}.jpg"
        save_screenshot(screenshot, str(path))

    return screenshot, width, height


async def _run_detection(
    detector: Detector,
    screenshot: bytes,
    iteration: int,
    alarm: bool,
) -> list[DetectedEntity]:
    """Run entity detection, choosing adaptive SAHI or standard mode."""
    try:
        if config.adaptive_sahi:
            force_full = iteration == 1 or iteration % config.full_sahi_interval == 0 or alarm
            entities = await _invoke_detector(
                detector,
                "detect_adaptive",
                screenshot,
                force_full=force_full,
            )
        else:
            # Single-pass. Use detect_fast: it is single-pass on BOTH the local and
            # remote detectors. (The remote detector's detect() maps to /detect/sahi,
            # which is bad for v6 — see Chapter 7 §7.4 — whereas detect_fast hits the
            # single-pass /detect endpoint at imgsz.)
            entities = await _invoke_detector(detector, "detect_fast", screenshot)
        set_detected_entities(entities)
        log.debug("detection_complete", entity_count=len(entities))
        return entities
    except Exception as e:
        log.warning("detection_failed", error=str(e))
        clear_detected_entities()
        return []


async def _classify_entities(
    detected_entities: list,
    screenshot: bytes,
) -> tuple[str, dict]:
    """Build entity summary and classify ownership of military units.

    The classifier is CPU-bound, so it runs in a thread: it was the last such
    step left on the event loop.
    """
    ownership_results: dict = {}
    if not detected_entities:
        return "", ownership_results

    try:
        from detection.inference.ownership import classify_entities as classify_ownership

        from .goals import THREAT_CLASSES

        ownership_results = await asyncio.to_thread(
            classify_ownership, screenshot, detected_entities, THREAT_CLASSES
        )
    except Exception:
        pass

    entity_summary = build_entity_summary(
        detected_entities,
        max_count=ENTITY_DISPLAY_LIMIT,
        ownership_results=ownership_results,
    )
    return entity_summary, ownership_results
