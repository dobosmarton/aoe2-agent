"""Trackers under comparison, all plugged into the real `sequence_replay`.

Each contender replaces `TemporalEntities` and `prepare_detections` inside
`sequence_replay` for one replay, the way the existing oracle test patches the tracker.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from gameplay_agent import sequence_replay, temporal_entities, tracking
from gameplay_agent.sequence_replay import _overlap

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import ModuleType

    from gameplay_agent.camera_motion import CameraMotion
    from gameplay_agent.entity_snapshot import EntitySnapshot

Box = tuple[float, float, float, float]
Prepared = temporal_entities.PreparedDetection

HISTORY_SECONDS = 1.0
IOU_MATCH = 0.3
ORACLE_MATCH = 0.5
SLOWDOWN_SECONDS = 0.03
NATIVE_BOX_LENGTH = 4


@dataclass(frozen=True, slots=True)
class FrameRecord:
    capture_id: int
    captured_at: float
    entities: tuple[EntitySnapshot, ...]
    prepare_ms: float
    associate_ms: float


class Tracker(Protocol):
    last_ambiguous_count: int
    last_candidate_ms: float
    last_matching_ms: float

    def accepts(self, capture_id: int, captured_at: float) -> bool: ...
    def observe(
        self,
        detections: tuple[Prepared, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]: ...


class Contender(Protocol):
    name: str

    def prepare(
        self, entities: Sequence[EntitySnapshot], pixels: np.ndarray
    ) -> tuple[Prepared, ...]: ...
    def tracker(self, directory: Path) -> Tracker: ...


def _stamp(
    entity: EntitySnapshot,
    identifier: str,
    label: str,
    agreement: float | None,
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


def _no_appearance(entities: Sequence[EntitySnapshot]) -> tuple[Prepared, ...]:
    return tuple(Prepared(item, ()) for item in entities if item.bbox is not None)


@dataclass(slots=True)
class _Baseline:
    """Shared bookkeeping for contenders that have no tracker statistics."""

    last_ambiguous_count: int = 0
    last_candidate_ms: float = 0.0
    last_matching_ms: float = 0.0
    latest_capture: int = -1
    latest_at: float = -1.0

    def accepts(self, capture_id: int, captured_at: float) -> bool:
        return capture_id > self.latest_capture and captured_at > self.latest_at


@dataclass(slots=True)
class RawTracker(_Baseline):
    """No tracking: every detection is new, and labels are the detector's."""

    next_id: int = 0

    def observe(
        self,
        detections: tuple[Prepared, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        self.latest_capture, self.latest_at = capture_id, captured_at
        output: list[EntitySnapshot] = []
        for item in detections:
            identifier = f"raw_{self.next_id}"
            self.next_id += 1
            entity = item.entity
            output.append(
                _stamp(
                    entity,
                    identifier,
                    entity.class_name,
                    None,
                    capture_id,
                    camera_revision,
                    captured_at,
                )
            )
        return tuple(output)


@dataclass(slots=True)
class OracleTracker(_Baseline):
    """Perfect identity and labels from ground truth, which validates the metrics."""

    truth: dict[int, list[tuple[str, str, Box]]] = field(default_factory=dict)
    next_id: int = 0

    def observe(
        self,
        detections: tuple[Prepared, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        self.latest_capture, self.latest_at = capture_id, captured_at
        pairs = sorted(
            (
                (_overlap(item.entity.bbox or (), box), index, identifier, label)
                for index, item in enumerate(detections)
                for identifier, label, box in self.truth.get(capture_id, [])
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        assigned: dict[int, tuple[str, str]] = {}
        used: set[str] = set()
        for overlap, index, identifier, label in pairs:
            if overlap < ORACLE_MATCH:
                break
            if index not in assigned and identifier not in used:
                assigned[index] = (identifier, label)
                used.add(identifier)
        output: list[EntitySnapshot] = []
        for index, item in enumerate(detections):
            entity = item.entity
            if index in assigned:
                identifier, label = assigned[index]
            else:
                identifier, label = f"false_{self.next_id}", entity.class_name
                self.next_id += 1
            output.append(
                _stamp(entity, identifier, label, None, capture_id, camera_revision, captured_at)
            )
        return tuple(output)


@dataclass(slots=True)
class _IouTrack:
    identifier: str
    box: Box
    last_seen_at: float


@dataclass(slots=True)
class IouTracker(_Baseline):
    """Greedy IoU matching with one second of memory and camera compensation."""

    tracks: list[_IouTrack] = field(default_factory=list)
    next_id: int = 0

    def observe(
        self,
        detections: tuple[Prepared, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        if motion.kind == "reset" or captured_at - self.latest_at > HISTORY_SECONDS:
            self.tracks = []
        self.latest_capture, self.latest_at = capture_id, captured_at
        dx, dy = motion.shift
        self.tracks = [
            _IouTrack(
                track.identifier,
                (track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy),
                track.last_seen_at,
            )
            for track in self.tracks
            if captured_at - track.last_seen_at <= HISTORY_SECONDS
        ]
        boxes = [cast("Box", item.entity.bbox) for item in detections]
        pairs = sorted(
            (
                (_overlap(track.box, box), track_index, box_index)
                for track_index, track in enumerate(self.tracks)
                for box_index, box in enumerate(boxes)
            ),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for overlap, track_index, box_index in pairs:
            if overlap < IOU_MATCH:
                break
            if box_index not in assigned and track_index not in used:
                assigned[box_index] = track_index
                used.add(track_index)
        output: list[EntitySnapshot] = []
        for box_index, item in enumerate(detections):
            if box_index in assigned:
                track = self.tracks[assigned[box_index]]
                track.box, track.last_seen_at = boxes[box_index], captured_at
            else:
                track = _IouTrack(f"iou_{self.next_id}", boxes[box_index], captured_at)
                self.next_id += 1
                self.tracks.append(track)
            entity = item.entity
            output.append(
                _stamp(
                    entity,
                    track.identifier,
                    entity.class_name,
                    None,
                    capture_id,
                    camera_revision,
                    captured_at,
                )
            )
        return tuple(output)


class Experiment(temporal_entities.TemporalEntities):
    """The `candidate` slot: override a method here to judge an idea by the same gates."""


def load_screentrack() -> ModuleType:
    """Import the native library, refusing a debug build that would ruin timings."""
    import screentrack

    if screentrack.__debug_build__:
        raise RuntimeError("screentrack is a debug build; timings would be meaningless")
    return screentrack


NativeTracker = tracking.NativeTemporalEntities


class _SlowedTracker(NativeTracker):
    """Injected regression: the gates must notice this."""

    def observe(
        self, detections: tuple[Prepared, ...], **kwargs: object
    ) -> tuple[EntitySnapshot, ...]:
        time.sleep(SLOWDOWN_SECONDS)
        return super().observe(detections, **kwargs)  # pyright: ignore[reportArgumentType]


def _truth(directory: Path) -> dict[int, list[tuple[str, str, Box]]]:
    frames: dict[int, list[tuple[str, str, Box]]] = {}
    for line in (directory / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frames[row["capture_id"]] = [
            (item["track_id"], item["class_name"], tuple(item["bbox"])) for item in row["objects"]
        ]
    return frames


@dataclass(frozen=True, slots=True)
class _NoAppearance:
    name: str

    def prepare(
        self, entities: Sequence[EntitySnapshot], pixels: np.ndarray
    ) -> tuple[Prepared, ...]:
        return _no_appearance(entities)

    def tracker(self, directory: Path) -> Tracker:
        if self.name == "oracle":
            return OracleTracker(truth=_truth(directory))
        if self.name == "raw":
            return RawTracker()
        return IouTracker()


@dataclass(frozen=True, slots=True)
class _PythonContender:
    name: str

    def prepare(
        self, entities: Sequence[EntitySnapshot], pixels: np.ndarray
    ) -> tuple[Prepared, ...]:
        return temporal_entities.prepare_detections(entities, pixels)

    def tracker(self, directory: Path) -> Tracker:
        if self.name == "candidate":
            return Experiment()
        return temporal_entities.TemporalEntities()


@dataclass(frozen=True, slots=True)
class _NativeContender:
    name: str

    def prepare(
        self, entities: Sequence[EntitySnapshot], pixels: np.ndarray
    ) -> tuple[Prepared, ...]:
        boxed = [item for item in entities if item.bbox is not None]
        boxes = np.array([item.bbox for item in boxed], dtype=np.float64).reshape(
            -1, NATIVE_BOX_LENGTH
        )
        signatures = load_screentrack().rgb_mean_std(pixels, boxes).tolist()
        return tuple(
            Prepared(item, tuple(signature))
            for item, signature in zip(boxed, signatures, strict=True)
        )

    def tracker(self, directory: Path) -> Tracker:
        return _SlowedTracker() if self.name == "slowed" else NativeTracker()


def contender(name: str) -> Contender:
    if name in {"raw", "oracle", "iou"}:
        return _NoAppearance(name)
    if name in {"screentrack", "slowed"}:
        return _NativeContender(name)
    if name in {"python", "python_again", "candidate"}:
        return _PythonContender(name)
    raise ValueError(f"unknown contender: {name}")


class _Recorder:
    """Stands in for `TemporalEntities`, timing and recording each frame."""

    def __init__(self, entrant: Contender, directory: Path) -> None:
        self._entrant = entrant
        self._tracker = entrant.tracker(directory)
        self._prepare_ms = 0.0
        self.frames: list[FrameRecord] = []

    @property
    def last_ambiguous_count(self) -> int:
        return self._tracker.last_ambiguous_count

    @property
    def last_candidate_ms(self) -> float:
        return self._tracker.last_candidate_ms

    @property
    def last_matching_ms(self) -> float:
        return self._tracker.last_matching_ms

    def accepts(self, capture_id: int, captured_at: float) -> bool:
        return self._tracker.accepts(capture_id, captured_at)

    def prepare(
        self, entities: Sequence[EntitySnapshot], pixels: np.ndarray
    ) -> tuple[Prepared, ...]:
        started = time.perf_counter()
        prepared = self._entrant.prepare(entities, pixels)
        self._prepare_ms = (time.perf_counter() - started) * 1000
        return prepared

    def observe(
        self,
        detections: tuple[Prepared, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        started = time.perf_counter()
        output = self._tracker.observe(
            detections,
            capture_id=capture_id,
            captured_at=captured_at,
            camera_revision=camera_revision,
            motion=motion,
        )
        associate_ms = (time.perf_counter() - started) * 1000
        self.frames.append(
            FrameRecord(capture_id, captured_at, output, self._prepare_ms, associate_ms)
        )
        return output


@contextmanager
def _patched(recorder: _Recorder) -> Iterator[None]:
    """Make the replay use this contender instead of the agent's chosen backend."""
    chosen = sequence_replay.make_tracking
    sequence_replay.make_tracking = lambda: tracking.Tracking("python", recorder.prepare, recorder)
    try:
        yield
    finally:
        sequence_replay.make_tracking = chosen


def replay(entrant: Contender, directory: Path) -> tuple[list[FrameRecord], dict[str, object]]:
    """Run one contender through the real replay, returning its frames and report."""
    recorder = _Recorder(entrant, directory)
    with _patched(recorder):
        report = sequence_replay.replay_sequence(
            directory, directory / "raw.jsonl", directory / "labels.jsonl"
        )
    return recorder.frames, report
