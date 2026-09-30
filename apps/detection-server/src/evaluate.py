"""Reproducible validation of the served ONNX detector on labeled real frames.

Run from the repository root with::

    uv run python -m detection_server.evaluate \
        --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
        --data packages/detection/src/training_data_v9_slim \
        --output packages/detection/registry/aoe2-entity-detector/evaluation.json

This scores images separately, not tracking or the frame-cache path. The
bundled validation split is not independent of the training capture sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import TYPE_CHECKING

from core import DetectedEntity
from detection.inference.postprocess import nms
from detection.inference.thresholds import (
    BUILDING_FALSE_POSITIVE_FLOORS,
    CLASS_THRESHOLDS,
    DEFAULT_CONFIDENCE,
)
from detection.labeling.class_mapping import load_classes_yaml
from detection.testing.evaluate_real import (
    GroundTruth,
    Prediction,
    Sample,
    Tally,
    _micro,
    _prf,
    _score,
)
from detection_server.app import ModelState, _detect_single, _load_class_names

if TYPE_CHECKING:
    from PIL import Image


MATCH_IOU = 0.5
INFERENCE_SIZE = 1280


@dataclass(frozen=True, slots=True)
class Case:
    image: Path
    label: Path


@dataclass(frozen=True, slots=True)
class Metrics:
    ground_truth: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def discover_cases(data_dir: Path) -> list[Case]:
    """Require one label for each real validation image."""
    images = data_dir / "val" / "images"
    labels = data_dir / "val" / "labels"
    cases = [
        Case(image=image, label=labels / f"{image.stem}.txt")
        for image in sorted(images.glob("real_*"))
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not cases:
        raise ValueError(f"No real validation images in {images}")
    missing = [case.label for case in cases if not case.label.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} labels, first: {missing[0]}")
    return cases


def load_ground_truth(label: Path, size: tuple[int, int], class_count: int) -> list[GroundTruth]:
    """Read strict YOLO labels; incomplete labels must not improve a score."""
    width, height = size
    annotations: list[GroundTruth] = []
    for line_number, line in enumerate(label.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label {label}:{line_number}")
        class_id = int(parts[0])
        cx, cy, box_width, box_height = (float(part) for part in parts[1:])
        values = (cx, cy, box_width, box_height)
        if not 0 <= class_id < class_count or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid class or coordinate {label}:{line_number}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f"Out-of-range box {label}:{line_number}")
        annotations.append(
            GroundTruth(
                class_id=class_id,
                box=(
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ),
            )
        )
    return annotations


def predict(state: ModelState, image: Image.Image) -> list[Prediction]:
    """Use the served single-pass decoder, thresholds, and client-side NMS."""
    detections = _detect_single(state, image, INFERENCE_SIZE)
    entities = [
        DetectedEntity(
            id=str(index),
            class_name=result.class_name,
            bbox=result.bbox,
            center=result.center,
            confidence=result.confidence,
            area=result.area,
        )
        for index, result in enumerate(detections)
    ]
    class_ids = {name: index for index, name in enumerate(state.class_names)}
    return [
        Prediction(
            class_id=class_ids[entity.class_name], box=entity.bbox, confidence=entity.confidence
        )
        for entity in nms(entities, iou_threshold=MATCH_IOU)
        if entity.class_name in class_ids
    ]


def collect_samples(state: ModelState, cases: list[Case]) -> list[Sample]:
    from PIL import Image as PILImage

    samples: list[Sample] = []
    for case in cases:
        with PILImage.open(case.image) as source:
            image = source.convert("RGB")
        ground_truth = load_ground_truth(case.label, image.size, len(state.class_names))
        samples.append((predict(state, image), ground_truth))
    return samples


def load_cpu_model(model: Path) -> ModelState:
    """Use a portable provider while retaining the server's decoder and thresholds."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return ModelState(
        backend="onnx_cpu",
        model=session,
        class_names=_load_class_names(),
        model_path=str(model),
        input_name=session.get_inputs()[0].name,
    )


def metrics_for(tally: Tally) -> Metrics:
    """Convert a scorer tally to a serializable fixed-operating-point result."""
    prf = _prf(tally)
    return Metrics(tally.gt, tally.tp, tally.fp, tally.fn, prf.precision, prf.recall, prf.f1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(cases: list[Case]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        for path in (case.image, case.label):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_report(
    model: Path, cases: list[Case], state: ModelState, samples: list[Sample]
) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[3]
    class_schema = repo / "packages/detection/src/training/config/classes.yaml"
    thresholds_source = repo / "packages/detection/src/inference/thresholds.py"
    server_source = Path(__file__).with_name("app.py")
    by_class = _score(samples, conf=0.0, iou_thr=MATCH_IOU)
    totals = _micro(by_class)
    thresholds = {
        "default": DEFAULT_CONFIDENCE,
        "class": CLASS_THRESHOLDS,
        "building_false_positive_floors": BUILDING_FALSE_POSITIVE_FLOORS,
    }
    return {
        "evaluation_kind": "in-distribution_real_frame_validation",
        "independent_test_set": False,
        "model_sha256": sha256_file(model),
        "class_schema_sha256": sha256_file(class_schema),
        "dataset_manifest_sha256": manifest_sha256(cases),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "server_source_sha256": sha256_file(server_source),
        "thresholds_source_sha256": sha256_file(thresholds_source),
        "runtime": {
            "python": python_version(),
            "onnxruntime": version("onnxruntime"),
            "pillow": version("Pillow"),
        },
        "images": len(cases),
        "ground_truth_boxes": totals.gt,
        "inference": {
            "path": "detection_server.app._detect_single + remote-client NMS",
            "backend": state.backend,
            "input_size": INFERENCE_SIZE,
            "matching_iou": MATCH_IOU,
            "thresholds": thresholds,
            "tracking": False,
        },
        "micro": asdict(metrics_for(totals)),
        "per_class": {
            name: asdict(metrics_for(by_class.get(index, Tally())))
            for index, name in enumerate(state.class_names)
        },
    }


class _Args(argparse.Namespace):
    model: Path
    data: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(namespace=_Args())

    model: Path = args.model
    data: Path = args.data
    output: Path = args.output
    if not model.is_file() or model.suffix != ".onnx":
        raise ValueError(f"Expected an ONNX model: {model}")
    cases = discover_cases(data)
    state = load_cpu_model(model)
    schema = load_classes_yaml()
    if tuple(schema[index] for index in sorted(schema)) != state.class_names:
        raise ValueError("Detector server and training class schemas disagree")
    samples = collect_samples(state, cases)
    report = build_report(model, cases, state, samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    micro_f1 = _prf(_micro(_score(samples, conf=0.0, iou_thr=MATCH_IOU))).f1
    print(f"Wrote {output}: {len(cases)} images; micro F1={micro_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
