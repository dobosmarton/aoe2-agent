"""Kalman filter-based multi-object tracker for AoE2 entities.

Replaces greedy IoU ID assignment with proper tracking:
- Hungarian algorithm for optimal detection-to-track matching
- Kalman filter for velocity estimation and position prediction
- Stable entity IDs across frames, even for fast-moving units
- Prediction-only mode for skipping rescans when confident

Time base: the agent's detection cadence is irregular — intra-turn rescans land
~0.3s apart while per-turn detections wait on an LLM round-trip (seconds). A
fixed one-step-per-call transition would therefore measure velocity in
"px per however long that call happened to take". The filter reads elapsed
wall-clock itself (via an injected `clock`) and rebuilds F and Q per step, so
velocities are px/second and comparable across ticks.

Camera motion is *not* modelled: a pan or zoom displaces every box at once,
which no constant-velocity model can express. Callers that know the camera
moved must call `reset()` — see `EntityTracker.reset`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from core import DetectedEntity
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class TrackedEntity:
    """An entity being tracked across frames."""

    id: str
    class_name: str
    state: np.ndarray  # [x_center, y_center, vx, vy, width, height], velocity in px/s
    covariance: np.ndarray  # 6x6 covariance matrix
    hits: int = 1  # consecutive successful matches
    misses: int = 0  # consecutive frames without a match
    confidence: float = 0.0  # last detection confidence


# Measurement matrix H: observe [x_center, y_center, width, height]
_H = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)

# Measurement noise R: tuned for YOLO bbox jitter (~3-5 px). Detector jitter does
# not depend on how long we waited, so unlike Q this stays constant.
_R = np.diag([5.0, 5.0, 3.0, 3.0]) ** 2

# Initial covariance for new tracks (high uncertainty on velocity)
_P0 = np.diag([10.0, 10.0, 100.0, 100.0, 10.0, 10.0]) ** 2

# Process noise expressed as *rates*, so Q can be rebuilt for whatever dt
# actually elapsed. Units start, stop and change direction abruptly, so the
# unmodelled acceleration is large.
_ACCEL_NOISE = 200.0  # px/s^2, position + velocity coupling
_SIZE_NOISE = 10.0  # px/s, bbox width/height drift (zoom, sprite animation)

# Beyond this gap the velocity estimate is stale: extrapolating further flings
# the box off-screen, the IoU drops to zero and the track dies anyway. Saturate
# and let the fresh detection carry the update instead.
_MAX_EXTRAPOLATION_S = 1.0

# Consecutive detections a track needs before its ID is believable. Fresh tracks
# (camera just moved, tracker just reset) have not earned an identity yet.
_MIN_HITS_TO_TRUST = 3


class EntityTracker:
    """Kalman filter-based multi-object tracker.

    Usage:
        tracker = EntityTracker()

        # Each detection cycle:
        stable_entities = tracker.update(raw_detections)

        # Between detection cycles (instant, no inference):
        predicted = tracker.predict()
        confidence = tracker.prediction_confidence()
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tracks: list[TrackedEntity] = []
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._clock = clock
        self._last_step = clock()
        self._next_id = 0

    def update(self, detections: list[DetectedEntity]) -> list[DetectedEntity]:
        """Match detections to tracks, update Kalman state, return with stable IDs."""
        dt = self._elapsed()
        for track in self.tracks:
            _kalman_predict(track, dt)

        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return self._tracks_to_entities()

        if not detections:
            for track in self.tracks:
                self._mark_missed(track)
            self._prune_dead_tracks()
            return self._tracks_to_entities()

        cost_matrix = self._match_costs(detections)
        track_indices, det_indices = _solve_assignment(cost_matrix)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for t_idx, d_idx in zip(track_indices, det_indices, strict=True):
            if cost_matrix[t_idx, d_idx] < (1.0 - self.iou_threshold):
                self._absorb_detection(self.tracks[t_idx], detections[d_idx])
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                self._mark_missed(track)

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        self._prune_dead_tracks()
        return self._tracks_to_entities()

    def predict(self) -> list[DetectedEntity]:
        """Predict current positions without new detections (instant).

        Use between detection cycles when `prediction_confidence()` is high.
        """
        dt = self._elapsed()
        for track in self.tracks:
            _kalman_predict(track, dt)
        return self._tracks_to_entities()

    def prediction_confidence(self) -> float:
        """How far tracked positions can be trusted without a detection (0-1).

        Fraction of tracks that are both currently matched *and* confirmed by
        `_MIN_HITS_TO_TRUST` consecutive detections. Match rate alone is not
        enough: every freshly created track is matched by construction, so a
        tracker repopulated right after a camera move would read ~1.0 at the
        exact moment it knows least about identity.
        """
        if not self.tracks:
            return 0.0
        trusted = sum(1 for t in self.tracks if t.misses == 0 and t.hits >= _MIN_HITS_TO_TRUST)
        return trusted / len(self.tracks)

    def reset(self) -> None:
        """Drop every track — prior identities are no longer valid.

        Call this whenever the camera pans or zooms. Every box shifts at once,
        which the constant-velocity model cannot represent. Re-minting IDs costs
        continuity; matching across the jump is worse, because a shifted frame
        of look-alike entities (villagers on a gold pile, a row of houses) has
        real IoU with the *neighbour* and hands each track the wrong identity
        silently.
        """
        self.tracks.clear()
        self._last_step = self._clock()

    def _elapsed(self) -> float:
        """Seconds since the previous step, saturated for long gaps."""
        now = self._clock()
        dt = now - self._last_step
        self._last_step = now
        return min(max(dt, 0.0), _MAX_EXTRAPOLATION_S)

    def _match_costs(self, detections: list[DetectedEntity]) -> np.ndarray:
        """Cost matrix (tracks x detections) of 1 - IoU; 1.0 blocks a pairing."""
        costs = np.ones((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            track_bbox = _state_to_bbox(track.state)
            for j, det in enumerate(detections):
                if track.class_name == det.class_name:
                    costs[i, j] = 1.0 - _iou(track_bbox, det.bbox)
        return costs

    def _create_track(self, detection: DetectedEntity) -> None:
        meas = _bbox_to_measurement(detection.bbox)
        state = np.array([meas[0], meas[1], 0.0, 0.0, meas[2], meas[3]])
        self.tracks.append(
            TrackedEntity(
                id=self._new_id(detection.class_name),
                class_name=detection.class_name,
                state=state,
                covariance=_P0.copy(),
                confidence=detection.confidence,
            )
        )

    def _absorb_detection(self, track: TrackedEntity, detection: DetectedEntity) -> None:
        _kalman_update(track, _bbox_to_measurement(detection.bbox))
        track.hits += 1
        track.misses = 0
        track.confidence = detection.confidence

    def _mark_missed(self, track: TrackedEntity) -> None:
        track.misses += 1
        track.hits = 0

    def _prune_dead_tracks(self) -> None:
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def _new_id(self, class_name: str) -> str:
        tid = f"{class_name}_{self._next_id}"
        self._next_id += 1
        return tid

    def _tracks_to_entities(self) -> list[DetectedEntity]:
        """Convert active tracks back to DetectedEntity list."""
        entities = []
        for track in self.tracks:
            if track.misses > 0:
                continue  # Only return actively tracked entities
            bbox = _state_to_bbox(track.state)
            entities.append(
                DetectedEntity(
                    id=track.id,
                    class_name=track.class_name,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    center=(
                        float(cast("float", track.state[0])),
                        float(cast("float", track.state[1])),
                    ),
                    confidence=track.confidence,
                    area=float(
                        max(1.0, float(cast("float", track.state[4])))
                        * max(1.0, float(cast("float", track.state[5])))
                    ),
                )
            )
        entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return entities


# --- Module-level helper functions ---


def _transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity transition F for a step of `dt` seconds."""
    f = np.eye(6)
    f[0, 2] = dt  # x += vx * dt
    f[1, 3] = dt  # y += vy * dt
    return f


def _process_noise(dt: float) -> np.ndarray:
    """Piecewise white-noise acceleration model for a step of `dt` seconds.

    An unmodelled acceleration `a` over `dt` displaces the entity by a*dt^2/2
    and changes its speed by a*dt — so position and velocity uncertainty grow
    together and are correlated. Bbox size is a plain random walk.
    """
    accel_var = _ACCEL_NOISE**2
    position = accel_var * dt**4 / 4
    position_velocity = accel_var * dt**3 / 2
    velocity = accel_var * dt**2
    size = (_SIZE_NOISE * dt) ** 2

    q = np.zeros((6, 6))
    q[0, 0] = q[1, 1] = position
    q[2, 2] = q[3, 3] = velocity
    q[0, 2] = q[2, 0] = position_velocity  # x <-> vx
    q[1, 3] = q[3, 1] = position_velocity  # y <-> vy
    q[4, 4] = q[5, 5] = size
    return q


def _kalman_predict(track: TrackedEntity, dt: float) -> None:
    """Kalman predict step: advance state `dt` seconds along its velocity."""
    f = _transition_matrix(dt)
    track.state = f @ track.state
    track.covariance = f @ track.covariance @ f.T + _process_noise(dt)


def _kalman_update(track: TrackedEntity, measurement: np.ndarray) -> None:
    """Kalman update step: correct prediction with measurement."""
    y = measurement - _H @ track.state
    S = _H @ track.covariance @ _H.T + _R
    K = track.covariance @ _H.T @ np.linalg.inv(S)
    track.state = track.state + K @ y
    track.covariance = (np.eye(6) - K @ _H) @ track.covariance


def _bbox_to_measurement(bbox: tuple) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to measurement [x_c, y_c, w, h]."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def _state_to_bbox(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert state [x_c, y_c, vx, vy, w, h] to (x1, y1, x2, y2)."""
    x_c = float(cast("float", state[0]))
    y_c = float(cast("float", state[1]))
    w = max(1.0, float(cast("float", state[4])))
    h = max(1.0, float(cast("float", state[5])))
    return (x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2)


def _iou(box1: tuple, box2: tuple) -> float:
    """Calculate IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _solve_assignment(cost_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """Solve the assignment problem (Hungarian algorithm)."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(row_ind), list(col_ind)
