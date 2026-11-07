"""Short visual history for fresh detector boxes on a moving game camera."""

from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from .camera_motion import MOBILE_CLASSES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from .camera_motion import CameraMotion
    from .entity_snapshot import EntitySnapshot

_HISTORY_SECONDS = 1.0
_HALF_LIFE_SECONDS = 0.4
_MAX_OBSERVATIONS = 3
_MIN_CLASS_MARGIN = 0.25
_AMBIGUITY_MARGIN = 0.15
_STALE_OVERLAP = 0.5
_LABEL_PENALTY = 1.0
_GAP_FACTOR = 3.0
# A new label is the detector changing its mind about one box, never a different box.
_RELABEL_OVERLAP = 0.5
# Buildings and resources don't move, so their tracks only continue on a box they touch.
# Low on purpose: the camera estimate can be a box-width off during a pan.
_STATIC_OVERLAP = 0.1
# They also outlive a missed box. A neighbour forgotten after a second came back as a
# stranger, and the track of a building missed at that moment took it.
_STATIC_MEMORY_SECONDS = 5.0


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
    crop = pixels[max(0, y0) : max(0, min(height, y1)), max(0, x0) : max(0, min(width, x1))]
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


def _box_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _last_label(track: _Track) -> str | None:
    return track.evidence[-1].label if track.evidence else None


def _is_unit(track: _Track) -> bool:
    return _last_label(track) in MOBILE_CLASSES


def _is_stale_duplicate(track: _Track, updated: Sequence[_Track]) -> bool:
    """An unmatched track on top of a fresh one would make that object ambiguous forever.

    Only the same label counts. The detector often puts two labels on one spot, and a
    second label that skips a frame should get its ID back when it returns.
    """
    label = _last_label(track)
    return any(
        _box_overlap(track.box, item.box) > _STALE_OVERLAP and _last_label(item) == label
        for item in updated
    )


def _matching_cost(
    track: _Track,
    box: tuple[float, float, float, float],
    appearance: tuple[float, ...],
    elapsed: float,
    label: str,
) -> float | None:
    """Rank a pair by overlap, looks and label, or refuse it.

    Overlap separates neighbours far better than distance, because a tracked object
    barely moves between frames. The label penalty stops a second label on one spot from
    tying with the first, while a flickering label can still continue its track.

    Only units may continue on a box they don't touch. Without that limit, a farm whose
    box was missed for a frame passed its ID to a villager standing in it, and the
    villager walked off with it.
    """
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
    overlap = _box_overlap(track.box, box)
    previous = _last_label(track)
    relabelled = previous is not None and previous != label
    if relabelled and overlap < _RELABEL_OVERLAP:
        return None
    units = previous in MOBILE_CLASSES and label in MOBILE_CLASSES
    if previous is not None and not units and overlap < _STATIC_OVERLAP:
        return None
    position = 1.0 - overlap if overlap > 0.0 else 1.0 + distance / allowed
    return position + appearance_distance + (_LABEL_PENALTY if relabelled else 0.0)


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
            cost = _matching_cost(
                track, box, detection.appearance, elapsed, detection.entity.class_name
            )
            if cost is not None:
                candidates.append((cost, old_index, new_index))
    candidates.sort()
    return candidates


def _check_runner_up(
    cost: float, identity: int, cheapest: dict[int, float | None], ambiguous: set[int]
) -> None:
    """Compare an identity's second candidate with its first, then ignore the rest."""
    if identity not in cheapest:
        cheapest[identity] = cost
        return
    best = cheapest[identity]
    if best is not None and cost - best < _AMBIGUITY_MARGIN:
        ambiguous.add(identity)
    cheapest[identity] = None


def _ambiguous_candidates(
    candidates: Sequence[tuple[float, int, int]],
) -> tuple[set[int], set[int]]:
    """Only an identity's two cheapest candidates can make it ambiguous.

    Candidates are sorted, so those are the first two seen for each identity. Worse
    options must not count: in a crowded frame nearly every identity has some pair of
    similar ones, which rejected almost every match on real footage.
    """
    cheapest_old: dict[int, float | None] = {}
    cheapest_new: dict[int, float | None] = {}
    ambiguous_old: set[int] = set()
    ambiguous_new: set[int] = set()
    for cost, old_index, new_index in candidates:
        _check_runner_up(cost, old_index, cheapest_old, ambiguous_old)
        _check_runner_up(cost, new_index, cheapest_new, ambiguous_new)
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
    if not total:
        return current, 0.0
    winner = max(scores, key=scores.__getitem__)
    agreement = scores[winner] / total
    if winner != current:
        runner_up = max((score for label, score in scores.items() if label != winner), default=0.0)
        supporting = sum(item.label == winner for item in evidence)
        if supporting < 2 or (scores[winner] - runner_up) / total < _MIN_CLASS_MARGIN:
            return current, scores.get(current, 0.0) / total
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
        if motion.kind == "reset":
            self.reset()
        # Slow frames still need history: a fixed lifetime would forget everything at once.
        lifetime = max(_HISTORY_SECONDS, _GAP_FACTOR * (captured_at - self._last_at))
        static_lifetime = max(lifetime, _STATIC_MEMORY_SECONDS)
        self._last_capture_id = capture_id
        self._last_at = captured_at
        fresh = tuple(item.entity for item in detections)
        tracks = [
            replace(item, box=_translated_box(item.box, motion.shift))
            for item in self._tracks
            if captured_at - item.last_seen_at <= (lifetime if _is_unit(item) else static_lifetime)
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
            *(
                track
                for index, track in enumerate(tracks)
                if index not in used_old and not _is_stale_duplicate(track, updated)
            ),
        )
        self.last_matching_ms = (time.perf_counter() - matching_started) * 1000
        return tuple(output)


__all__ = ["PreparedDetection", "TemporalEntities", "prepare_detections"]
