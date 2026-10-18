"""Short visual history for fresh detector boxes on a moving game camera."""

from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from .camera_motion import CameraMotion
    from .entity_snapshot import EntitySnapshot

_HISTORY_SECONDS = 1.0
_HALF_LIFE_SECONDS = 0.4
_MAX_OBSERVATIONS = 3
_MIN_CLASS_MARGIN = 0.25


@dataclass(frozen=True, slots=True)
class _Evidence:
    label: str
    confidence: float
    captured_at: float


@dataclass(frozen=True, slots=True)
class _Track:
    id: str
    box: tuple[float, float, float, float]
    appearance: tuple[float, ...]
    evidence: tuple[_Evidence, ...]
    last_seen_at: float


@dataclass(frozen=True, slots=True)
class PreparedDetection:
    entity: EntitySnapshot
    appearance: tuple[float, ...]


def _appearance(pixels: np.ndarray, box: tuple[float, ...]) -> tuple[float, ...]:
    height, width = cast("tuple[int, int]", pixels.shape[:2])
    x0, y0, x1, y1 = (int(value) for value in box)
    crop = pixels[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    if crop.size == 0:
        return (0.0,) * 6
    # NumPy's installed stubs expose aggregate arrays as Any; narrow here.
    mean_array = cast("np.ndarray", crop.mean(axis=(0, 1)) / 255.0)
    std_array = cast("np.ndarray", crop.std(axis=(0, 1)) / 255.0)
    mean = cast("list[float]", mean_array.tolist())
    std = cast("list[float]", std_array.tolist())
    return (*mean, *std)


def prepare_detections(
    entities: Sequence[EntitySnapshot], pixels: np.ndarray
) -> tuple[PreparedDetection, ...]:
    """Read each current box once from one immutable capture view."""
    return tuple(
        PreparedDetection(item, _appearance(pixels, item.bbox))
        for item in entities
        if item.bbox is not None
    )


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _translated_box(
    box: tuple[float, float, float, float], shift: tuple[float, float]
) -> tuple[float, float, float, float]:
    return (box[0] + shift[0], box[1] + shift[1], box[2] + shift[0], box[3] + shift[1])


def _matching_cost(
    track: _Track,
    box: tuple[float, float, float, float],
    appearance: tuple[float, ...],
    elapsed: float,
) -> float | None:
    old_x, old_y = _box_center(track.box)
    new_x, new_y = _box_center(box)
    distance = math.hypot(new_x - old_x, new_y - old_y)
    size = max(track.box[2] - track.box[0], track.box[3] - track.box[1], 12.0)
    allowed = size * 0.75 + 65.0 * elapsed
    if distance > allowed:
        return None
    appearance_distance = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(track.appearance, appearance, strict=True))
    )
    if appearance_distance > 0.55:
        return None
    return distance / allowed + appearance_distance


def _candidates(
    tracks: Sequence[_Track], detections: Sequence[PreparedDetection], captured_at: float
) -> list[tuple[float, int, int]]:
    """Generate the same feasible pairs, pruning by horizontal movement first."""
    indexed = sorted(
        (_box_center(item.entity.bbox)[0], index)
        for index, item in enumerate(detections)
        if item.entity.bbox is not None
    )
    centers = [center for center, _index in indexed]
    candidates: list[tuple[float, int, int]] = []
    for old_index, track in enumerate(tracks):
        old_x, _old_y = _box_center(track.box)
        elapsed = captured_at - track.last_seen_at
        size = max(track.box[2] - track.box[0], track.box[3] - track.box[1], 12.0)
        allowed = size * 0.75 + 65.0 * elapsed
        for _center, new_index in indexed[
            bisect_left(centers, old_x - allowed) : bisect_right(centers, old_x + allowed)
        ]:
            detection = detections[new_index]
            box = detection.entity.bbox
            assert box is not None
            cost = _matching_cost(track, box, detection.appearance, elapsed)
            if cost is not None:
                candidates.append((cost, old_index, new_index))
    candidates.sort()
    return candidates


def _ambiguous_candidates(
    candidates: Sequence[tuple[float, int, int]],
) -> tuple[set[int], set[int]]:
    """Adjacent costs within each identity suffice because candidates are sorted."""
    previous_old: dict[int, float] = {}
    previous_new: dict[int, float] = {}
    ambiguous_old: set[int] = set()
    ambiguous_new: set[int] = set()
    for cost, old_index, new_index in candidates:
        old_cost = previous_old.get(old_index)
        new_cost = previous_new.get(new_index)
        if old_cost is not None and cost - old_cost < 0.15:
            ambiguous_old.add(old_index)
        if new_cost is not None and cost - new_cost < 0.15:
            ambiguous_new.add(new_index)
        previous_old[old_index] = cost
        previous_new[new_index] = cost
    return ambiguous_old, ambiguous_new


def _matches(
    candidates: Sequence[tuple[float, int, int]],
    ambiguous_old: set[int],
    ambiguous_new: set[int],
) -> tuple[dict[int, int], set[int]]:
    matched: dict[int, int] = {}
    used_old: set[int] = set()
    for _cost, old_index, new_index in candidates:
        if (
            new_index not in ambiguous_new
            and old_index not in ambiguous_old
            and new_index not in matched
            and old_index not in used_old
        ):
            matched[new_index] = old_index
            used_old.add(old_index)
    return matched, used_old


def _label(evidence: tuple[_Evidence, ...], current: str) -> tuple[str, float | None]:
    if len(evidence) < 2:
        return current, None
    now = evidence[-1].captured_at
    scores: dict[str, float] = {}
    for item in evidence:
        weight = item.confidence * 2 ** (-(now - item.captured_at) / _HALF_LIFE_SECONDS)
        scores[item.label] = scores.get(item.label, 0.0) + weight
    total = sum(scores.values())
    winner = max(scores, key=scores.__getitem__)
    agreement = scores[winner] / total if total else 0.0
    if winner != current:
        runner_up = max((score for label, score in scores.items() if label != winner), default=0.0)
        supporting = sum(item.label == winner for item in evidence)
        if supporting < 2 or (scores[winner] - runner_up) / total < _MIN_CLASS_MARGIN:
            return current, scores.get(current, 0.0) / total if total else 0.0
    return winner, agreement


class TemporalEntities:
    """One per game. All updates are ordered by capture time on the event loop."""

    def __init__(self) -> None:
        self._tracks: tuple[_Track, ...] = ()
        self._next_id = 0
        self._last_capture_id = -1
        self._last_at = -1.0
        self.last_ambiguous_count = 0
        self.last_candidate_ms = 0.0
        self.last_matching_ms = 0.0

    def reset(self) -> None:
        self._tracks = ()

    def accepts(self, capture_id: int, captured_at: float) -> bool:
        """Do not replay or publish a detector result older than the last one."""
        return capture_id > self._last_capture_id and captured_at > self._last_at

    def observe(
        self,
        detections: tuple[PreparedDetection, ...],
        *,
        capture_id: int,
        captured_at: float,
        camera_revision: int,
        motion: CameraMotion,
    ) -> tuple[EntitySnapshot, ...]:
        if not self.accepts(capture_id, captured_at):
            return ()
        self.last_ambiguous_count = 0
        if motion.kind == "reset" or captured_at - self._last_at > _HISTORY_SECONDS:
            self.reset()
        self._last_capture_id = capture_id
        self._last_at = captured_at
        fresh = tuple(item.entity for item in detections)
        tracks = [
            replace(item, box=_translated_box(item.box, motion.shift))
            for item in self._tracks
            if captured_at - item.last_seen_at <= _HISTORY_SECONDS
        ]
        candidate_started = time.perf_counter()
        candidates = _candidates(tracks, detections, captured_at)
        self.last_candidate_ms = (time.perf_counter() - candidate_started) * 1000
        matching_started = time.perf_counter()
        ambiguous_old, ambiguous_new = _ambiguous_candidates(candidates)
        self.last_ambiguous_count = len(ambiguous_new)
        matched, used_old = _matches(candidates, ambiguous_old, ambiguous_new)
        output: list[EntitySnapshot] = []
        updated: list[_Track] = []
        for index, item in enumerate(fresh):
            assert item.bbox is not None
            old = tracks[matched[index]] if index in matched else None
            if old is None:
                track_id = f"track_{self._next_id}"
                self._next_id += 1
                history: tuple[_Evidence, ...] = ()
            else:
                track_id = old.id
                history = old.evidence
            evidence = (*history, _Evidence(item.class_name, item.confidence, captured_at))
            evidence = tuple(
                value for value in evidence if captured_at - value.captured_at <= _HISTORY_SECONDS
            )[-_MAX_OBSERVATIONS:]
            label, agreement = _label(evidence, item.class_name)
            output.append(
                replace(
                    item,
                    id=track_id,
                    class_name=label,
                    raw_class_name=item.class_name,
                    class_agreement=agreement,
                    capture_id=capture_id,
                    camera_revision=camera_revision,
                    captured_at=captured_at,
                )
            )
            updated.append(
                _Track(track_id, item.bbox, detections[index].appearance, evidence, captured_at)
            )
        self._tracks = (
            *updated,
            *(track for index, track in enumerate(tracks) if index not in used_old),
        )
        self.last_matching_ms = (time.perf_counter() - matching_started) * 1000
        return tuple(output)


__all__ = ["PreparedDetection", "TemporalEntities", "prepare_detections"]
