"""The release score is strict about labels and uses the deployed NMS path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from detection.testing.evaluate_real import GroundTruth, Prediction
from detection_server.app import DetectionResult, ModelState
from detection_server.benchmark_latency import percentile
from detection_server.evaluate import (
    Case,
    build_report,
    collect_samples,
    discover_cases,
    fine_label_recall,
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


def test_only_the_test_split_claims_independence(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    image = tmp_path / "real_frame.jpg"
    label = tmp_path / "real_frame.txt"
    model.write_bytes(b"model")
    image.write_bytes(b"image")
    label.write_text("")
    state = ModelState("onnx_cpu", object(), ("tree",), str(model))

    report = build_report(model, [Case(image, label)], state, [([], [])], independent_test=True)

    assert report["independent_test_set"] is True
    assert report["evaluation_kind"] == "independent_real_frame_test"


def test_classes_a_model_cannot_predict_count_as_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "real_frame.png"
    label = tmp_path / "real_frame.txt"
    Image.new("RGB", (100, 100)).save(image)
    label.write_text("62 0.5 0.5 0.2 0.2\n")  # buffalo: appended after the 60-class v9
    state = ModelState("onnx_cpu", object(), ("tree", "villager"), "model.onnx")
    monkeypatch.setattr("detection_server.evaluate._detect_single", lambda *_: [])

    samples = collect_samples(state, [Case(image, label)])
    report = build_report(image, [Case(image, label)], state, samples)

    assert report["micro"]["false_negative"] == 1
    assert report["per_class"]["buffalo"]["ground_truth"] == 1


def test_latency_percentiles_interpolate_between_samples() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.5) == 25.0
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.95) == 38.5


@pytest.mark.parametrize("samples, fraction", [([], 0.5), ([1.0], -0.1), ([1.0], 1.1)])
def test_latency_percentile_rejects_invalid_input(samples: list[float], fraction: float) -> None:
    with pytest.raises(ValueError):
        percentile(samples, fraction)


def test_fine_label_recall_counts_hits_on_the_mapped_class(tmp_path: Path) -> None:
    fine = tmp_path / "real_frame.fine.txt"
    fine.write_text("ibex\nibex\ndeer\n")
    case = Case(tmp_path / "real_frame.jpg", tmp_path / "real_frame.txt", fine)
    ground_truth = [
        GroundTruth(5, (10, 10, 30, 30)),
        GroundTruth(5, (50, 50, 70, 70)),
        GroundTruth(5, (80, 10, 99, 30)),
    ]
    predictions = [Prediction(5, (10, 10, 30, 30), 0.9), Prediction(60, (50, 50, 70, 70), 0.8)]

    report = fine_label_recall([case], [(predictions, ground_truth)])

    assert report["ibex"] == {
        "maps_to": "deer",
        "ground_truth": 2,
        "true_positive": 1,
        "recall": 0.5,
    }
    assert report["elephant"]["ground_truth"] == 0
