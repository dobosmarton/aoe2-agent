"""Choose the tracker: screentrack's native one by default, or the Python reference.

`AOE2_TRACKER` selects the backend:

- `native` (the default): screentrack computes appearance signatures and tracks.
- `python`: the reference in `temporal_entities.py`.
- `shadow`: both run on the same evidence. The agent acts on Python's output and logs
  every frame where the two disagree, so a real game can check the native tracker.

If screentrack can't be imported, `native` falls back to Python with a warning.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import structlog

from .camera_motion import MOBILE_CLASSES
from .temporal_entities import PreparedDetection, TemporalEntities, prepare_detections

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from screentrack import Tracker as NativeLibraryTracker

    from .camera_motion import CameraMotion
    from .entity_snapshot import EntitySnapshot

log = structlog.stdlib.get_logger()

BACKEND_ENV = "AOE2_TRACKER"
Backend = Literal["native", "python", "shadow"]
_BACKENDS: tuple[Backend, ...] = ("native", "python", "shadow")
_BOX_LENGTH = 4
_SIGNATURE_LENGTH = 6


class Tracker(Protocol):
    """What the perception loop needs from a tracker."""

    @property
    def last_ambiguous_count(self) -> int: ...
    @property
    def last_candidate_ms(self) -> float: ...
    @property
    def last_matching_ms(self) -> float: ...
    def accepts(self, capture_id: int, captured_at: float) -> bool: ...
    def reset(self) -> None: ...
    def observe(
        self,
        detections: tuple[PreparedDetection, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]: ...


@dataclass(frozen=True, slots=True)
class Tracking:
    """A matched pair: the signatures a tracker expects, and the tracker."""

    backend: Backend
    prepare: Callable[[Sequence[EntitySnapshot], np.ndarray], tuple[PreparedDetection, ...]]
    tracker: Tracker


def native_prepare(
    entities: Sequence[EntitySnapshot], pixels: np.ndarray
) -> tuple[PreparedDetection, ...]:
    """Signatures for every box in one native call, matching `prepare_detections`."""
    import screentrack

    boxed = [item for item in entities if item.bbox is not None]
    boxes = np.array([item.bbox for item in boxed], dtype=np.float64).reshape(-1, _BOX_LENGTH)
    signatures = screentrack.rgb_mean_std(np.ascontiguousarray(pixels), boxes).tolist()
    return tuple(
        PreparedDetection(item, tuple(signature))
        for item, signature in zip(boxed, signatures, strict=True)
    )


def _stamp(
    entity: EntitySnapshot,
    identifier: str,
    label: str,
    agreement: float | None,
    *,
    capture_id: int,
    camera_revision: int,
    captured_at: float,
) -> EntitySnapshot:
    return replace(
        entity,
        id=identifier,
        class_name=label,
        raw_class_name=entity.class_name,
        class_agreement=agreement,
        capture_id=capture_id,
        camera_revision=camera_revision,
        captured_at=captured_at,
    )


class NativeTemporalEntities:
    """`TemporalEntities` backed by screentrack, translating snapshots to arrays."""

    def __init__(self) -> None:
        import screentrack

        self._tracker: NativeLibraryTracker = screentrack.Tracker(
            screentrack.Config.aoe2(), MOBILE_CLASSES
        )

    @property
    def last_ambiguous_count(self) -> int:
        return self._tracker.ambiguous_count

    @property
    def last_candidate_ms(self) -> float:
        return self._tracker.candidate_ms

    @property
    def last_matching_ms(self) -> float:
        return self._tracker.matching_ms

    def accepts(self, capture_id: int, captured_at: float) -> bool:
        return self._tracker.accepts(capture_id, captured_at)

    def reset(self) -> None:
        self._tracker.reset()

    def observe(
        self,
        detections: tuple[PreparedDetection, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        fresh = [item.entity for item in detections]
        shift = ("pan", motion.shift[0], motion.shift[1]) if motion.kind == "pan" else motion.kind
        result = self._tracker.observe(
            capture_id=capture_id,
            captured_at=captured_at,
            motion=shift,
            boxes=np.array([item.bbox for item in fresh], dtype=np.float64).reshape(
                -1, _BOX_LENGTH
            ),
            labels=[item.class_name for item in fresh],
            confidences=np.array([item.confidence for item in fresh], dtype=np.float64),
            signatures=np.array([item.appearance for item in detections], dtype=np.float64).reshape(
                -1, _SIGNATURE_LENGTH
            ),
        )
        if result is None:
            return ()
        track_ids, labels, agreements = result
        return tuple(
            _stamp(
                item,
                f"track_{track_id}",
                label,
                None if math.isnan(agreement) else agreement,
                capture_id=capture_id,
                camera_revision=camera_revision,
                captured_at=captured_at,
            )
            for item, track_id, label, agreement in zip(
                fresh, track_ids.tolist(), labels, agreements.tolist(), strict=True
            )
        )


def _decisions(output: tuple[EntitySnapshot, ...]) -> list[tuple[str, str]]:
    return [(item.id, item.class_name) for item in output]


class ShadowTemporalEntities:
    """Runs both trackers on the same evidence, returns Python's and logs differences."""

    def __init__(self) -> None:
        self._python = TemporalEntities()
        self._native = NativeTemporalEntities()
        self.frames = 0
        self.differing_frames = 0

    @property
    def last_ambiguous_count(self) -> int:
        return self._python.last_ambiguous_count

    @property
    def last_candidate_ms(self) -> float:
        return self._python.last_candidate_ms

    @property
    def last_matching_ms(self) -> float:
        return self._python.last_matching_ms

    def accepts(self, capture_id: int, captured_at: float) -> bool:
        return self._python.accepts(capture_id, captured_at)

    def reset(self) -> None:
        self._python.reset()
        self._native.reset()

    def observe(
        self,
        detections: tuple[PreparedDetection, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        python = self._python.observe(
            detections,
            capture_id=capture_id,
            captured_at=captured_at,
            camera_revision=camera_revision,
            motion=motion,
        )
        native = self._native.observe(
            detections,
            capture_id=capture_id,
            captured_at=captured_at,
            camera_revision=camera_revision,
            motion=motion,
        )
        self.frames += 1
        if _decisions(python) != _decisions(native):
            self.differing_frames += 1
            log.warning(
                "tracker_shadow_differs",
                capture_id=capture_id,
                python=_decisions(python),
                native=_decisions(native),
                differing_frames=self.differing_frames,
                frames=self.frames,
            )
        return python


def _requested(value: str | None) -> Backend:
    for backend in _BACKENDS:
        if value == backend:
            return backend
    if value:
        log.warning("tracker_backend_unknown", requested=value, using="native")
    return "native"


def _native_available() -> bool:
    try:
        import screentrack
    except ImportError:
        return False
    if screentrack.__debug_build__:
        log.warning("tracker_native_debug_build", version=screentrack.__version__)
    return True


def make_tracking(backend: str | None = None) -> Tracking:
    """The tracker the environment asks for, falling back to Python if needed."""
    chosen = _requested(os.environ.get(BACKEND_ENV) if backend is None else backend)
    if chosen != "python" and not _native_available():
        log.warning("tracker_native_unavailable", requested=chosen, using="python")
        chosen = "python"
    if chosen == "native":
        tracking = Tracking(chosen, native_prepare, NativeTemporalEntities())
    elif chosen == "shadow":
        tracking = Tracking(chosen, prepare_detections, ShadowTemporalEntities())
    else:
        tracking = Tracking(chosen, prepare_detections, TemporalEntities())
    log.info("tracker_backend", backend=tracking.backend)
    return tracking


__all__ = [
    "BACKEND_ENV",
    "Backend",
    "NativeTemporalEntities",
    "ShadowTemporalEntities",
    "Tracker",
    "Tracking",
    "make_tracking",
    "native_prepare",
]
