"""Raw and temporal scoring consume the same cached detections."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from core import DetectedEntity
from gameplay_agent.capture_sequence import record_sequence
from gameplay_agent.screen import NativeCapture
from gameplay_agent.sequence_replay import cache_raw_detections, replay_sequence

if TYPE_CHECKING:
    from pathlib import Path


def _sequence(tmp_path: Path) -> Path:
    capture_id = 0

    def grab() -> NativeCapture:
        nonlocal capture_id
        at = 1.0 + capture_id * 0.2
        capture_id += 1
        return NativeCapture(bytes(64 * 36 * 3), 64, 36, at, (0, 0, 64, 36))

    directory = tmp_path / "sequence"
    record_sequence(directory, count=2, interval=0.2, grab=grab, sleep=lambda _s: None)
    return directory


def test_replay_scores_cached_detections_and_tracks_identity(tmp_path: Path) -> None:
    directory = _sequence(tmp_path)
    detections = tmp_path / "detections.jsonl"
    labels = tmp_path / "labels.jsonl"
    box = [10, 10, 20, 20]
    detections.write_text(
        "\n".join(
            json.dumps(
                {
                    "capture_id": index,
                    "detections": [{"class_name": "sheep", "bbox": box, "confidence": 0.9}],
                }
            )
            for index in range(2)
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            json.dumps(
                {
                    "capture_id": index,
                    "objects": [{"class_name": "sheep", "bbox": box, "track_id": "sheep_1"}],
                }
            )
            for index in range(2)
        ),
        encoding="utf-8",
    )

    report = replay_sequence(directory, detections, labels)

    assert report["frames"] == 2
    assert report["raw"]["micro"]["f1"] == 1.0
    assert report["temporal"]["micro"]["f1"] == 1.0
    assert report["raw"]["identity_switches"] is None
    assert report["temporal"]["identity_switches"] == 0
    for stage in (
        "camera",
        "preparation",
        "candidate",
        "matching",
        "association",
        "temporal_total",
    ):
        assert report[f"{stage}_ms_p50"] is not None
        assert report[f"{stage}_ms_p95"] is not None

    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(
        '{"capture_id": 0, "target_class": "sheep", "x": 15, "y": 15}\n'
        '{"capture_id": 1, "target_class": "sheep", "x": 30, "y": 30}\n',
        encoding="utf-8",
    )
    assert replay_sequence(directory, detections, labels, attempts)["wrong_target_attempts"] == 1


def test_replay_requires_reviewed_labels_for_every_frame(tmp_path: Path) -> None:
    directory = _sequence(tmp_path)
    detections = tmp_path / "detections.jsonl"
    labels = tmp_path / "labels.jsonl"
    detections.write_text(
        '{"capture_id": 0, "detections": []}\n{"capture_id": 1, "detections": []}\n',
        encoding="utf-8",
    )
    labels.write_text('{"capture_id": 0, "objects": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="capture 1"):
        replay_sequence(directory, detections, labels)


def test_raw_detections_are_inferred_once_then_cached(tmp_path: Path) -> None:
    directory = _sequence(tmp_path)
    output = tmp_path / "raw.jsonl"

    class FakeDetector:
        calls = 0

        def detect_fast(self, image: bytes) -> list[DetectedEntity]:
            assert image.startswith(b"\x89PNG")
            self.calls += 1
            return [DetectedEntity("raw", "sheep", (10, 10, 20, 20), (15, 15), 0.9)]

    detector = FakeDetector()
    cache_raw_detections(directory, output, detector)
    assert detector.calls == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(FileExistsError):
        cache_raw_detections(directory, output, detector)
    assert detector.calls == 2
