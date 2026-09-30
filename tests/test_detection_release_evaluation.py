"""The release score is strict about labels and uses the deployed NMS path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from detection.testing.evaluate_real import GroundTruth, Prediction
from detection_server.app import DetectionResult, ModelState
from detection_server.evaluate import (
    Case,
    build_report,
    discover_cases,
    load_ground_truth,
    manifest_sha256,
    predict,
)
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_real_validation_label_fails_closed(tmp_path: Path) -> None:
    images = tmp_path / "val/images"
    images.mkdir(parents=True)
    (images / "real_frame.jpg").write_bytes(b"image")

    with pytest.raises(ValueError, match="Missing 1 labels"):
        discover_cases(tmp_path)


@pytest.mark.parametrize(
    "contents",
    ["0 0.5 0.5 0.2", "60 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"],
)
def test_malformed_ground_truth_is_rejected(tmp_path: Path, contents: str) -> None:
    label = tmp_path / "frame.txt"
    label.write_text(contents)

    with pytest.raises(ValueError):
        load_ground_truth(label, (100, 100), class_count=60)


def test_manifest_changes_when_a_label_changes(tmp_path: Path) -> None:
    image = tmp_path / "real_frame.jpg"
    label = tmp_path / "real_frame.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    case = Case(image, label)
    before = manifest_sha256([case])

    label.write_text("0 0.6 0.5 0.2 0.2\n")

    assert manifest_sha256([case]) != before


def test_prediction_applies_the_remote_clients_classwise_nms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ModelState("onnx_cpu", object(), ("tree", "villager"), "model.onnx")
    results = [
        DetectionResult(
            class_name=name,
            bbox=(10, 10, 30, 30),
            center=(20, 20),
            confidence=confidence,
            area=400,
        )
        for name, confidence in (("tree", 0.9), ("tree", 0.7), ("villager", 0.8))
    ]
    monkeypatch.setattr("detection_server.evaluate._detect_single", lambda *_: results)

    predictions = predict(state, Image.new("RGB", (100, 100)))

    assert [(item.class_id, item.confidence) for item in predictions] == [(0, 0.9), (1, 0.8)]


def test_report_includes_false_positives_and_model_hash(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    image = tmp_path / "real_frame.jpg"
    label = tmp_path / "real_frame.txt"
    model.write_bytes(b"model")
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    ground_truth = GroundTruth(0, (10, 10, 30, 30))
    samples = [
        (
            [
                Prediction(0, (10, 10, 30, 30), 0.9),
                Prediction(1, (40, 40, 60, 60), 0.9),
            ],
            [ground_truth],
        )
    ]
    state = ModelState("onnx_cpu", object(), ("tree", "villager"), str(model))

    report = build_report(model, [Case(image, label)], state, samples)

    assert report["independent_test_set"] is False
    assert report["images"] == 1
    assert report["ground_truth_boxes"] == 1
    assert report["micro"]["true_positive"] == 1
    assert report["micro"]["false_positive"] == 1
    assert report["micro"]["f1"] == pytest.approx(2 / 3)
    assert len(report["model_sha256"]) == 64
