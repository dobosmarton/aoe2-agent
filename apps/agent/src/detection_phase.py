"""Vision-pipeline glue between the game loop and the YOLO detector.

Owns:
  - `init_detector` / `init_frame_differ`: optional-resource initialization
    (the agent can run without a detector).
  - `detect_frame`: one ladder from cheapest to costliest — tracker prediction,
    pan translation, then a real detection.
  - `_capture_screenshot` / `summarize_frame`: the screenshot and the
    ownership tagging either side of it.

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
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from detection.inference.detector import DetectedEntity, EntityDetector
    from detection.inference.frame_diff import FrameChange, FrameDiffer
    from detection.inference.ownership import Owner
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
)
from .screen import capture_screenshot, save_screenshot

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


def init_detector() -> Detector | None:
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
        # Explicit: get_detector defaults to use_sahi=True, and SAHI tiles the
        # 3024 px frame into crops the model never trained on. Real F1 0.04 vs
        # 0.42 single-pass. v9 gets its pixels from imgsz=1280 instead.
        detector = get_detector(
            use_mock=False,
            imgsz=config.detection_imgsz,
            use_sahi=False,
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


def init_frame_differ() -> FrameDiffer | None:
    """Initialize frame differ for skipping redundant rescans."""
    try:
        from detection.inference.frame_diff import FrameDiffer

        return FrameDiffer(threshold=FRAME_DIFFER_THRESHOLD)
    except ImportError:
        return None


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


async def detect_frame(
    detector: Detector,
    differ: FrameDiffer | None,
    screenshot: bytes,
) -> Sequence[object]:
    """The cheapest view of this frame that is still true.

    Run 2026_08_22_1 served 112 of 212 frames from the 2 rungs above the
    detector. Every rung publishes to the entity cache `target_class` reads.
    """
    change = differ.compare(screenshot) if differ else None
    if change is None:
        return await _detected_frame(detector, screenshot)
    if not change.changed:
        return _unchanged_frame(detector)
    if config.rescan_cache:
        translated = _panned_frame(change)
        if translated is not None:
            return translated
    return await _detected_frame(detector, screenshot)


def _unchanged_frame(detector: Detector) -> Sequence[object]:
    """Nothing moved on screen. Extrapolate, or keep what we had."""
    tracker = detector.tracker
    if tracker and tracker.get_confidence() > TRACKER_CONFIDENCE_THRESHOLD:
        predicted = tracker.predict()
        set_detected_entities(predicted)
        log.debug("frame_predicted", entity_count=len(predicted))
        return predicted
    held = get_detected_entities()
    log.debug("frame_unchanged", entity_count=len(held))
    return held


def _panned_frame(change: FrameChange) -> Sequence[object] | None:
    """The view only panned, so shift the static map instead of re-detecting."""
    translated, declined = _pan_translation(change, get_detected_entities())
    if not translated:
        log.debug("frame_pan_declined", reason=declined, response=round(change.response, 3))
        return None
    set_detected_entities(translated)
    log.debug(
        "frame_translated",
        entity_count=len(translated),
        shift=[round(v) for v in change.shift],
    )
    return translated


async def _detected_frame(detector: Detector, screenshot: bytes) -> Sequence[object]:
    """Pay for a detection, and own the catch: this is the only rung that fails
    for a reason outside the process — no model, or a server that is down.

    `detect_fast` is single-pass on both detectors. The remote's `detect()`
    maps to /detect/sahi, which is bad for v6."""
    # The cache, not `detector._previous_entities`: after a cheap rung the cache
    # holds the current view, and it is public.
    previous = get_detected_entities()
    try:
        entities = await _invoke_detector(detector, "detect_fast", screenshot)
    except Exception as e:
        log.warning("detection_failed", error=str(e))
        clear_detected_entities()
        return []
    if detector.tracker and previous and len(entities) < len(previous) * ENTITY_DROP_RATIO:
        detector.tracker.reset()
        log.debug("tracker_reset", reason="camera_moved")
    set_detected_entities(entities)
    log.debug("frame_detected", entity_count=len(entities))
    return entities


async def summarize_frame(
    detected_entities: Sequence[object],
    screenshot: bytes,
) -> tuple[str, dict[str, tuple[Owner, float]]]:
    """The frame as the LLM reads it: a summary line, plus who owns each unit.

    The ownership classifier is CPU-bound, so it runs in a thread. It was the
    last such step left on the event loop.
    """
    ownership_results: dict[str, tuple[Owner, float]] = {}
    if not detected_entities:
        return "", ownership_results

    try:
        from detection.inference.ownership import classify_entities as classify_ownership

        from .goals import THREAT_CLASSES

        ownership_results = await asyncio.to_thread(
            classify_ownership, screenshot, list(detected_entities), THREAT_CLASSES
        )
    except Exception as e:
        # Ownership is an enrichment: the summary is still worth returning.
        log.debug("ownership_classification_failed", error=str(e))

    entity_summary = build_entity_summary(
        detected_entities,
        max_count=ENTITY_DISPLAY_LIMIT,
        ownership_results=ownership_results,
    )
    return entity_summary, ownership_results


__all__ = [
    "DETECTION_AVAILABLE",
    "ENTITY_DISPLAY_LIMIT",
    "STATIC_CLASSES",
    "detect_frame",
    "init_detector",
    "init_frame_differ",
    "summarize_frame",
]
