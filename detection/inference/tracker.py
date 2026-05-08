"""Kalman filter-based multi-object tracker for AoE2 entities.

Replaces greedy IoU ID assignment with proper tracking:
- Hungarian algorithm for optimal detection-to-track matching
- Kalman filter for velocity estimation and position prediction
- Stable entity IDs across frames, even for fast-moving units
- Prediction-only mode for skipping rescans when confident
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .detector import DetectedEntity

logger = logging.getLogger(__name__)


@dataclass
class TrackedEntity:
    """An entity being tracked across frames."""
    id: str
    class_name: str
    state: np.ndarray          # [x_center, y_center, vx, vy, width, height]
    covariance: np.ndarray     # 6x6 covariance matrix
    hits: int = 1              # consecutive successful matches
    misses: int = 0            # consecutive frames without a match
    confidence: float = 0.0    # last detection confidence


# Kalman filter constants (shared across all tracks)

# Transition matrix F: constant velocity model (x += vx, y += vy)
_F = np.array([
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=np.float64)

# Measurement matrix H: observe [x_center, y_center, width, height]
_H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=np.float64)

# Process noise Q: tuned for AoE2 unit speeds (~5-20 px/frame)
_Q = np.diag([10.0, 10.0, 5.0, 5.0, 2.0, 2.0]) ** 2

# Measurement noise R: tuned for YOLO bbox jitter (~3-5 px)
_R = np.diag([5.0, 5.0, 3.0, 3.0]) ** 2

# Initial covariance for new tracks (high uncertainty on velocity)
_P0 = np.diag([10.0, 10.0, 100.0, 100.0, 10.0, 10.0]) ** 2


class EntityTracker:
    """Kalman filter-based multi-object tracker.

    Usage:
        tracker = EntityTracker()

        # Each detection cycle:
        stable_entities = tracker.update(raw_detections)

        # Between detection cycles (instant, no inference):
        predicted = tracker.predict()
        confidence = tracker.get_confidence()
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 3):
        self.tracks: list[TrackedEntity] = []
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_id = 0

    def _new_id(self, class_name: str) -> str:
        tid = f"{class_name}_{self._next_id}"
        self._next_id += 1
        return tid

    def _create_track(self, detection: DetectedEntity):
        meas = _bbox_to_measurement(detection.bbox)
        state = np.array([meas[0], meas[1], 0.0, 0.0, meas[2], meas[3]])
        track = TrackedEntity(
            id=self._new_id(detection.class_name),
            class_name=detection.class_name,
            state=state,
            covariance=_P0.copy(),
            confidence=detection.confidence,
        )
        self.tracks.append(track)

    def update(self, detections: list[DetectedEntity]) -> list[DetectedEntity]:
        """Match detections to tracks, update Kalman state, return with stable IDs."""
        # 1. Predict step for all existing tracks
        for track in self.tracks:
            _kalman_predict(track)

        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return self._tracks_to_entities()

        if not detections:
            for track in self.tracks:
                track.misses += 1
                track.hits = 0
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
            return self._tracks_to_entities()

        # 2. Build cost matrix (num_tracks x num_detections) using 1 - IoU
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            track_bbox = _state_to_bbox(track.state)
            for j, det in enumerate(detections):
                if track.class_name != det.class_name:
                    continue  # Leave at 1.0 (max cost) for class mismatch
                cost_matrix[i, j] = 1.0 - _iou(track_bbox, det.bbox)

        # 3. Optimal assignment
        track_indices, det_indices = _solve_assignment(cost_matrix)

        # 4. Process matches
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for t_idx, d_idx in zip(track_indices, det_indices, strict=False):
            if cost_matrix[t_idx, d_idx] < (1.0 - self.iou_threshold):
                meas = _bbox_to_measurement(detections[d_idx].bbox)
                _kalman_update(self.tracks[t_idx], meas)
                self.tracks[t_idx].hits += 1
                self.tracks[t_idx].misses = 0
                self.tracks[t_idx].confidence = detections[d_idx].confidence
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

        # 5. Unmatched tracks: increment misses
        for i in range(n_tracks):
            if i not in matched_tracks:
                self.tracks[i].misses += 1
                self.tracks[i].hits = 0

        # 6. Unmatched detections: create new tracks
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(detections[j])

        # 7. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return self._tracks_to_entities()

    def predict(self) -> list[DetectedEntity]:
        """Predict current positions without new detections (instant).

        Use between detection cycles when tracker confidence is high.
        """
        for track in self.tracks:
            _kalman_predict(track)
        return self._tracks_to_entities()

    def get_confidence(self) -> float:
        """Overall tracker confidence (0-1).

        Returns ratio of actively matched tracks to total tracks.
        Low confidence = many lost/unmatched tracks.
        """
        if not self.tracks:
            return 0.0
        active = sum(1 for t in self.tracks if t.misses == 0)
        return active / len(self.tracks)

    def reset(self):
        """Clear all tracks (e.g., on camera movement)."""
        self.tracks.clear()

    def _tracks_to_entities(self) -> list[DetectedEntity]:
        """Convert active tracks back to DetectedEntity list."""
        entities = []
        for track in self.tracks:
            if track.misses > 0:
                continue  # Only return actively tracked entities
            bbox = _state_to_bbox(track.state)
            entities.append(DetectedEntity(
                id=track.id,
                class_name=track.class_name,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                center=(float(track.state[0]), float(track.state[1])),
                confidence=track.confidence,
                area=float(max(1, track.state[4]) * max(1, track.state[5])),
            ))
        entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return entities


# --- Module-level helper functions ---

def _kalman_predict(track: TrackedEntity):
    """Kalman predict step: advance state using constant velocity model."""
    track.state = _F @ track.state
    track.covariance = _F @ track.covariance @ _F.T + _Q


def _kalman_update(track: TrackedEntity, measurement: np.ndarray):
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
    x_c, y_c = state[0], state[1]
    w, h = max(1.0, state[4]), max(1.0, state[5])
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
    """Solve the assignment problem (Hungarian algorithm with greedy fallback)."""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(row_ind), list(col_ind)
    except ImportError:
        return _greedy_match(cost_matrix)


def _greedy_match(cost_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """Greedy matching fallback when scipy is not installed."""
    n_rows, n_cols = cost_matrix.shape
    row_indices: list[int] = []
    col_indices: list[int] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    # Flatten and sort by cost
    costs = []
    for i in range(n_rows):
        for j in range(n_cols):
            costs.append((cost_matrix[i, j], i, j))
    costs.sort()

    for _, r, c in costs:
        if r in used_rows or c in used_cols:
            continue
        row_indices.append(r)
        col_indices.append(c)
        used_rows.add(r)
        used_cols.add(c)

    return row_indices, col_indices
