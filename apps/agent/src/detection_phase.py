"""Single-frame detection and ownership at the gameplay boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from core import DetectedEntity
    from detection.inference.detector import EntityDetector
    from detection.inference.ownership import Owner
    from detection.inference.remote_detector import RemoteDetector

    from .game_profile import GameProfile

    Detector: TypeAlias = EntityDetector | RemoteDetector

from .config import config
from .entity_snapshot import EntitySnapshot, snapshot_entities
from .entity_utils import build_entity_summary
from .executor import GATE_BUILDING_CLASSES

log = structlog.stdlib.get_logger()


class DetectionUnavailableError(RuntimeError):
    """The configured detector cannot serve a recorded game."""


class DetectionBusyError(RuntimeError):
    """The detector is healthy but at capacity for this frame."""


_REMOTE_DETECTION_ERROR: type[RuntimeError] = DetectionUnavailableError
_REMOTE_BUSY_ERROR: type[RuntimeError] = DetectionBusyError
try:
    from detection.inference.detector import get_detector
    from detection.inference.remote_detector import (
        DetectionBusyError as RemoteDetectionBusyError,
    )
    from detection.inference.remote_detector import (
        DetectionUnavailableError as RemoteDetectionUnavailableError,
    )

    _REMOTE_DETECTION_ERROR = RemoteDetectionUnavailableError
    _REMOTE_BUSY_ERROR = RemoteDetectionBusyError
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    log.info("detection_not_available", message="Running without YOLO detection")


ENTITY_DISPLAY_LIMIT = 20


async def _invoke_detector(
    det: Detector, method: str, *args: object, **kwargs: object
) -> list[DetectedEntity]:
    """Call a detector method, handling both sync and async implementations.

    All EntityDetector / RemoteDetector inference methods return
    `list[DetectedEntity]` — pyright can't see that through `getattr`,
    so the return type is asserted here.
    """
    fn = cast("Callable[..., object]", getattr(det, method))
    try:
        if asyncio.iscoroutinefunction(fn):
            return cast("list[DetectedEntity]", await fn(*args, **kwargs))
        return cast("list[DetectedEntity]", await asyncio.to_thread(fn, *args, **kwargs))
    except _REMOTE_BUSY_ERROR as exc:
        raise DetectionBusyError(str(exc)) from exc
    except _REMOTE_DETECTION_ERROR as exc:
        raise DetectionUnavailableError(str(exc)) from exc


class DetectorFactory(Protocol):
    """Construct the detector policy required by one execution environment."""

    def __call__(self) -> Detector | None: ...


def init_detector() -> Detector | None:
    """Initialize a resilient interactive detector."""
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
        log.warning("detector_init_failed", error_type=type(e).__name__, error=repr(e))
        return None


def init_required_detector() -> Detector | None:
    """Initialize the configured detector or fail a recorded experiment."""
    if not DETECTION_AVAILABLE:
        raise DetectionUnavailableError("detection package is unavailable")
    try:
        if config.detection_host:
            from detection.inference.remote_detector import get_required_remote_detector

            detector = get_required_remote_detector(
                config.detection_host,
                imgsz=config.detection_imgsz,
            )
            log.info("detector_initialized", mode="required_remote", server=config.detection_host)
            return detector
        detector = get_detector(
            use_mock=False,
            imgsz=config.detection_imgsz,
            use_sahi=False,
            model_name=config.detection_model,
        )
        if detector.use_mock:
            raise DetectionUnavailableError(
                f"configured local model {config.detection_model!r} is unavailable"
            )
        log.info(
            "detector_initialized",
            mode=detector.backend or "yolo",
            confidence_threshold=detector.confidence_threshold,
        )
        return detector
    except DetectionUnavailableError:
        raise
    except Exception as exc:
        raise DetectionUnavailableError(
            f"could not initialize configured detector: {type(exc).__name__}: {exc!r}"
        ) from exc


async def detect_frame(
    detector: Detector,
    screenshot: bytes,
) -> tuple[EntitySnapshot, ...]:
    """Always run one real detector inference for the supplied capture."""
    try:
        entities = await _invoke_detector(detector, "detect_fast", screenshot)
    except DetectionBusyError:
        raise
    except DetectionUnavailableError:
        raise
    except Exception as e:
        log.warning("detection_failed", error_type=type(e).__name__, error=repr(e))
        return ()
    log.debug("frame_detected", entity_count=len(entities))
    return snapshot_entities(entities)


async def summarize_frame(
    detected_entities: Sequence[object],
    screenshot: bytes,
    *,
    profile: GameProfile | None = None,
) -> tuple[str, dict[str, tuple[Owner, float]]]:
    """The frame as the LLM reads it: a summary line, plus who owns each unit.

    The ownership classifier is CPU-bound, so it runs in a thread. It was the
    last such step left on the event loop.
    """
    ownership_results: dict[str, tuple[Owner, float]] = {}
    if not detected_entities:
        return "", ownership_results

    try:
        from detection.inference.ownership import Owner
        from detection.inference.ownership import classify_entities as classify_ownership

        from .goals import THREAT_CLASSES

        color_owners = (
            {player.color: Owner(player.relationship) for player in profile.players}
            if profile is not None and profile.roster_verified and profile.ownership_verified
            else None
        )
        ownership_results = await asyncio.to_thread(
            classify_ownership,
            screenshot,
            list(detected_entities),
            THREAT_CLASSES | GATE_BUILDING_CLASSES,
            color_owners,
        )
    except Exception as e:
        # Ownership is an enrichment: the summary is still worth returning.
        log.debug("ownership_classification_failed", error=str(e))

    entity_summary = build_entity_summary(
        list(detected_entities),
        max_count=ENTITY_DISPLAY_LIMIT,
        ownership_results=ownership_results,
    )
    return entity_summary, ownership_results


__all__ = [
    "DETECTION_AVAILABLE",
    "ENTITY_DISPLAY_LIMIT",
    "DetectionBusyError",
    "DetectionUnavailableError",
    "DetectorFactory",
    "detect_frame",
    "init_detector",
    "init_required_detector",
    "summarize_frame",
]
