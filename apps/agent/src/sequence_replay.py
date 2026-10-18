"""Score cached single-frame detections and temporal output on a reviewed sequence.

Inputs are JSONL beside a `capture_sequence` directory. Each detection line is
``{"capture_id": 0, "detections": [{"class_name": "sheep", "bbox": [..],
"confidence": 0.8}]}``. Labels use ``objects`` with a stable ``track_id``.
The replay never calls a detector; the same raw boxes feed both score paths.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from .camera_motion import (
    MOBILE_CLASSES,
    MotionLimits,
    analysis_pixels,
    cursor_exclusions,
    estimate_motion,
)
from .entity_snapshot import EntitySnapshot
from .temporal_entities import TemporalEntities, prepare_detections

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core import DetectedEntity


class SingleFrameDetector(Protocol):
    def detect_fast(self, screenshot: bytes) -> list[DetectedEntity]: ...


class _Frame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    capture_id: int
    captured_at_monotonic: float
    image: str
    input_events: tuple[str, ...] = ()
    cursor: tuple[int, int] | None = None


class _Box(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    track_id: str = ""


class _Detections(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    capture_id: int
    detections: tuple[_Box, ...]


class _Labels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    capture_id: int
    objects: tuple[_Box, ...]


class _Attempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    capture_id: int
    target_class: str
    x: float
    y: float


_Document = TypeVar("_Document", bound=BaseModel)


def _read_jsonl(path: Path, model: type[_Document]) -> tuple[_Document, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def cache_raw_detections(
    directory: Path,
    output: Path,
    detector: SingleFrameDetector,
) -> None:
    """Infer each sequence image once; retain raw boxes for repeatable replay."""
    if output.exists():
        raise FileExistsError(output)
    lines: list[str] = []
    for frame in _read_jsonl(directory / "sequence.jsonl", _Frame):
        if Path(frame.image).name != frame.image:
            raise ValueError("sequence images must be relative basenames")
        entities = detector.detect_fast((directory / frame.image).read_bytes())
        lines.append(
            json.dumps(
                {
                    "capture_id": frame.capture_id,
                    "detections": [
                        {
                            "class_name": entity.class_name,
                            "bbox": entity.bbox,
                            "confidence": entity.confidence,
                        }
                        for entity in entities
                    ],
                }
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0


@dataclass(slots=True)
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def report(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }


@dataclass(slots=True)
class _Scores:
    per_class: dict[str, _Counts] = field(default_factory=dict)
    label_flips: int = 0
    identity_switches: int = 0
    previous: dict[str, tuple[int, str, str]] = field(default_factory=dict)

    def score(
        self,
        frame_id: int,
        predictions: Sequence[EntitySnapshot],
        labels: Sequence[_Box],
    ) -> None:
        matched: set[int] = set()
        for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
            best = max(
                (
                    (_overlap(prediction.bbox, label.bbox), index)
                    for index, label in enumerate(labels)
                    if index not in matched
                    and prediction.bbox is not None
                    and prediction.class_name == label.class_name
                ),
                default=(0.0, -1),
            )
            if best[0] >= 0.5:
                self.per_class.setdefault(prediction.class_name, _Counts()).tp += 1
                matched.add(best[1])
            else:
                self.per_class.setdefault(prediction.class_name, _Counts()).fp += 1
        for index, label in enumerate(labels):
            if index not in matched:
                self.per_class.setdefault(label.class_name, _Counts()).fn += 1
            if not label.track_id:
                continue
            nearest = max(
                (
                    (_overlap(prediction.bbox, label.bbox), prediction)
                    for prediction in predictions
                    if prediction.bbox is not None
                ),
                key=lambda entry: entry[0],
                default=None,
            )
            if nearest is None or nearest[0] < 0.5:
                continue
            prediction = nearest[1]
            previous = self.previous.get(label.track_id)
            if previous is not None and previous[0] == frame_id - 1:
                self.label_flips += previous[1] != prediction.class_name
                self.identity_switches += previous[2] != prediction.id
            self.previous[label.track_id] = (frame_id, prediction.class_name, prediction.id)

    def report(self) -> dict[str, object]:
        micro = _Counts()
        for counts in self.per_class.values():
            micro.tp += counts.tp
            micro.fp += counts.fp
            micro.fn += counts.fn
        return {
            "micro": micro.report(),
            "per_class": {
                label: counts.report() for label, counts in sorted(self.per_class.items())
            },
            "label_flips": self.label_flips,
            "identity_switches": self.identity_switches,
        }


def _snapshot(box: _Box, identifier: str) -> EntitySnapshot:
    x0, y0, x1, y1 = box.bbox
    return EntitySnapshot(
        identifier,
        box.class_name,
        box.bbox,
        ((x0 + x1) / 2, (y0 + y1) / 2),
        box.confidence,
    )


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(len(ordered) * 0.95) - 1], 2)


def _p50(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def replay_sequence(
    directory: Path,
    detections_path: Path,
    labels_path: Path,
    attempts_path: Path | None = None,
) -> dict[str, object]:
    """Replay raw boxes once, scoring raw and camera-aware outputs identically."""
    frames = _read_jsonl(directory / "sequence.jsonl", _Frame)
    detections = {row.capture_id: row for row in _read_jsonl(detections_path, _Detections)}
    labels = {row.capture_id: row for row in _read_jsonl(labels_path, _Labels)}
    tracks = TemporalEntities()
    raw_scores, temporal_scores = _Scores(), _Scores()
    previous_analysis: np.ndarray | None = None
    previous_at = float("-inf")
    previous_boxes: tuple[tuple[float, float, float, float], ...] = ()
    previous_cursor: tuple[int, int] | None = None
    camera_times: list[float] = []
    preparation_times: list[float] = []
    association_times: list[float] = []
    candidate_times: list[float] = []
    matching_times: list[float] = []
    temporal_times: list[float] = []
    ambiguous = resets = pans = 0
    for frame in frames:
        if frame.capture_id not in detections or frame.capture_id not in labels:
            raise ValueError(
                f"missing detections or reviewed labels for capture {frame.capture_id}"
            )
        if Path(frame.image).name != frame.image:
            raise ValueError("sequence images must be relative basenames")
        with Image.open(directory / frame.image) as opened:
            image = opened.convert("RGB")
        rgb = image.tobytes()
        camera_started = time.monotonic()
        analysis = analysis_pixels(rgb, *image.size, MotionLimits())
        if frame.captured_at_monotonic - previous_at > 1.0 or any(
            event in {"camera_jump", "zoom"} for event in frame.input_events
        ):
            previous_analysis = None
        raw = tuple(
            _snapshot(box, f"raw_{frame.capture_id}_{index}")
            for index, box in enumerate(detections[frame.capture_id].detections)
        )
        current_boxes = tuple(
            item.bbox for item in raw if item.class_name in MOBILE_CLASSES and item.bbox is not None
        )
        limits = MotionLimits()
        motion = estimate_motion(
            previous_analysis,
            analysis,
            full_size=image.size,
            limits=limits,
            excluded_boxes=(
                *previous_boxes,
                *current_boxes,
                *cursor_exclusions(previous_cursor, limits),
                *cursor_exclusions(frame.cursor, limits),
            ),
        )
        camera_times.append((time.monotonic() - camera_started) * 1000)
        resets += motion.kind == "reset"
        pans += motion.kind == "pan"
        preparation_started = time.monotonic()
        prepared = prepare_detections(raw, np.asarray(image))
        prepared_ms = (time.monotonic() - preparation_started) * 1000
        preparation_times.append(prepared_ms)
        association_started = time.monotonic()
        temporal = tracks.observe(
            prepared,
            capture_id=frame.capture_id,
            captured_at=frame.captured_at_monotonic,
            camera_revision=resets + pans,
            motion=motion,
        )
        association_ms = (time.monotonic() - association_started) * 1000
        association_times.append(association_ms)
        candidate_times.append(tracks.last_candidate_ms)
        matching_times.append(tracks.last_matching_ms)
        temporal_times.append(prepared_ms + association_ms)
        ambiguous += tracks.last_ambiguous_count
        truth = labels[frame.capture_id].objects
        raw_scores.score(frame.capture_id, raw, truth)
        temporal_scores.score(frame.capture_id, temporal, truth)
        previous_analysis = analysis
        previous_at = frame.captured_at_monotonic
        previous_boxes = current_boxes
        previous_cursor = frame.cursor
    raw_report = raw_scores.report()
    raw_report["identity_switches"] = None  # Raw detections have no persistent IDs.
    wrong_targets: int | None = None
    if attempts_path is not None:
        wrong_targets = 0
        for attempt in _read_jsonl(attempts_path, _Attempt):
            reviewed = labels.get(attempt.capture_id)
            if reviewed is None:
                raise ValueError(f"missing reviewed labels for attempt {attempt.capture_id}")
            if not any(
                item.class_name == attempt.target_class
                and item.bbox[0] <= attempt.x <= item.bbox[2]
                and item.bbox[1] <= attempt.y <= item.bbox[3]
                for item in reviewed.objects
            ):
                wrong_targets += 1
    return {
        "frames": len(frames),
        "raw": raw_report,
        "temporal": temporal_scores.report(),
        "ambiguous_associations": ambiguous,
        "camera_resets": resets,
        "camera_pans": pans,
        "camera_ms_p50": _p50(camera_times),
        "camera_ms_p95": _p95(camera_times),
        "preparation_ms_p50": _p50(preparation_times),
        "preparation_ms_p95": _p95(preparation_times),
        "candidate_ms_p50": _p50(candidate_times),
        "candidate_ms_p95": _p95(candidate_times),
        "matching_ms_p50": _p50(matching_times),
        "matching_ms_p95": _p95(matching_times),
        "association_ms_p50": _p50(association_times),
        "association_ms_p95": _p95(association_times),
        "temporal_total_ms_p50": _p50(temporal_times),
        "temporal_total_ms_p95": _p95(temporal_times),
        "wrong_target_attempts": wrong_targets,
    }


class _Args(argparse.Namespace):
    sequence: Path
    detections: Path
    labels: Path
    attempts: Path | None
    infer: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, help="Optional reviewed click-attempt JSONL")
    parser.add_argument(
        "--infer", action="store_true", help="Write --detections once using local model weights"
    )
    args = parser.parse_args(namespace=_Args())
    if args.infer:
        from detection.inference.detector import get_detector

        from .config import config

        detector = get_detector(
            use_mock=False,
            use_sahi=False,
            imgsz=config.detection_imgsz,
            model_name=config.detection_model,
        )
        if detector.use_mock:
            raise RuntimeError("real detector weights are unavailable")
        cache_raw_detections(args.sequence, args.detections, detector)
    print(
        json.dumps(
            replay_sequence(args.sequence, args.detections, args.labels, args.attempts),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
