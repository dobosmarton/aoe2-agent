"""
Active learning pipeline for AoE2 object detection.

Systematically selects the most informative images for manual labeling,
so each CVAT labeling session maximally improves the model.

Workflow:
    1. Triage: Score all unlabeled images by informativeness
    2. Prepare: Create a CVAT-importable batch of the top-N images (with pre-labels)
    3. (Human labels in CVAT)
    4. Integrate: Import corrected labels into training dataset
    5. Retrain: Train new model on expanded dataset
    6. Repeat from step 1

Usage:
    # Score and rank all unlabeled images
    python -m detection.labeling.active_learning triage

    # Prepare next batch of 20 images for CVAT
    python -m detection.labeling.active_learning prepare --batch-size 20

    # Integrate corrected CVAT export back into training set
    python -m detection.labeling.active_learning integrate --cvat-export /path/to/export
"""

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

try:
    import PIL  # noqa: F401  # availability probe; PIL is used indirectly via other modules
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

from detection.inference._ultralytics_results import yolo_boxes_to_lists

from .class_mapping import load_classes_yaml, write_classes_txt

# Paths
_DETECTION_DIR = Path(__file__).parent.parent
_DEFAULT_MODEL = _DETECTION_DIR / "inference" / "models" / "aoe2_yolo_v6.pt"
_DEFAULT_RAW_DIR = _DETECTION_DIR / "real_screenshots" / "raw"
_DEFAULT_OUTPUT_DIR = _DETECTION_DIR / "labeling" / "output" / "active_learning"
_TRAINING_DATA_DIR = _DETECTION_DIR / "training_data"


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One raw detection from the triage pass (model class-name space)."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TriageItem:
    """An image's informativeness score plus the raw detections behind it."""

    path: str
    name: str
    score: int
    n_detections: int
    n_uncertain: int
    n_low: int
    n_high: int = 0
    reason: str = ""
    detections: tuple[DetectionRecord, ...] = ()


def triage(
    model_path: Path = _DEFAULT_MODEL,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    conf_low: float = 0.15,
    conf_high: float = 0.7,
) -> list[TriageItem]:
    """Score all images by informativeness for active learning.

    Images with many uncertain or missing detections are most valuable
    to label because they teach the model the most.

    Scoring:
        - Detections with conf < conf_low: +3 points (model is clueless)
        - Detections with conf_low <= conf < conf_high: +2 (model is uncertain)
        - No detections at all: +15 points (image may contain novel content)
        - Fewer total detections than expected: +5 (probably missing objects)

    Returns:
        Items sorted by score (most informative first). Each carries its raw
        detections so downstream tools (e.g. hard-negative mining) can reuse them.
    """
    try:
        from detection._ultralytics_compat import YOLO
    except ImportError:
        print("ERROR: ultralytics is required.")
        sys.exit(1)

    model = YOLO(str(model_path))
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in image_extensions])

    print(f"Triaging {len(images)} images...")

    scored: list[TriageItem] = []
    for i, img_path in enumerate(images):
        results = model(str(img_path), conf=0.05, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            scored.append(_empty_triage_item(img_path))
            continue

        detections = _to_detection_records(boxes, results[0].names)
        confs = [d.confidence for d in detections]
        n_low = sum(1 for c in confs if c < conf_low)
        n_uncertain = sum(1 for c in confs if conf_low <= c < conf_high)
        n_high = sum(1 for c in confs if c >= conf_high)

        # Uncertain and low-confidence detections are most informative; a sparse
        # image probably hides objects the model missed entirely.
        score = n_low * 3 + n_uncertain * 2
        if len(confs) < 5:
            score += 5

        scored.append(
            TriageItem(
                path=str(img_path),
                name=img_path.name,
                score=score,
                n_detections=len(confs),
                n_uncertain=n_uncertain,
                n_low=n_low,
                n_high=n_high,
                detections=detections,
            )
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(images)}] processed")

    scored.sort(key=lambda item: -item.score)

    output_dir.mkdir(parents=True, exist_ok=True)
    triage_path = output_dir / "triage_results.json"
    triage_path.write_text(json.dumps([asdict(item) for item in scored], indent=2) + "\n")

    print(f"\nTriage complete. Results saved to {triage_path}")
    print("\nTop 10 most informative images:")
    for item in scored[:10]:
        print(
            f"  Score {item.score:3d} | {item.n_detections:3d} det "
            f"({item.n_uncertain} uncertain, {item.n_low} low) | {item.name}"
        )

    print("\nBottom 5 (least informative):")
    for item in scored[-5:]:
        print(
            f"  Score {item.score:3d} | {item.n_detections:3d} det "
            f"({item.n_high} high-conf) | {item.name}"
        )

    return scored


def _empty_triage_item(img_path: Path) -> TriageItem:
    """A maximally-informative item for an image the model found nothing in."""
    return TriageItem(
        path=str(img_path),
        name=img_path.name,
        score=15,
        n_detections=0,
        n_uncertain=0,
        n_low=0,
        reason="no_detections",
    )


def _to_detection_records(boxes: object, names: object) -> tuple[DetectionRecord, ...]:
    """Convert an ultralytics `Boxes` object into typed detection records."""
    bboxes, class_ids, confidences = yolo_boxes_to_lists(boxes)
    class_names = cast("dict[int, str]", names)
    return tuple(
        DetectionRecord(
            class_name=class_names[int(class_id)],
            confidence=confidence,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        )
        for bbox, class_id, confidence in zip(bboxes, class_ids, confidences, strict=True)
    )


def prepare_batch(
    model_path: Path = _DEFAULT_MODEL,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    batch_size: int = 20,
    conf_threshold: float = 0.25,
) -> Path:
    """Prepare the next batch of most-informative images for CVAT labeling.

    Runs triage if not already done, then creates a CVAT-importable
    directory with images and pre-labels.

    Returns:
        Path to the batch directory.
    """
    try:
        from detection._ultralytics_compat import YOLO
    except ImportError:
        print("ERROR: ultralytics is required.")
        sys.exit(1)

    output_dir = Path(output_dir)

    # Resolve the top-N image paths from cached triage or a fresh run. Only the
    # ordering matters here; pre-labels are regenerated below by the model.
    triage_path = output_dir / "triage_results.json"
    if triage_path.exists():
        print("Loading existing triage results...")
        cached = cast("list[dict[str, object]]", json.loads(triage_path.read_text()))
        batch_paths = [Path(str(item["path"])) for item in cached[:batch_size]]
    else:
        print("No triage results found, running triage...")
        fresh = triage(model_path, raw_dir, output_dir)
        batch_paths = [Path(item.path) for item in fresh[:batch_size]]

    print(f"\nPreparing batch of {len(batch_paths)} images...")

    # Create batch directory
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    batch_dir = output_dir / f"batch_{timestamp}"
    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load model for pre-labeling
    model = YOLO(str(model_path))

    # Output classes (classes.yaml; the model emits these IDs natively)
    output_classes = load_classes_yaml()

    # Write classes.txt
    write_classes_txt(batch_dir / "classes.txt")

    # Process each image
    for img_path in batch_paths:
        if not img_path.exists():
            continue

        # Copy image
        dest_img = images_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Generate pre-labels
        results = model(str(img_path), conf=conf_threshold, verbose=False)

        labels = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

            for box, cls_id, _conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.cls,
                results[0].boxes.conf,
                strict=True,
            ):
                class_id = int(cls_id.item())
                if class_id not in output_classes:
                    continue

                x1, y1, x2, y2 = box.tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write label file
        label_name = img_path.stem + ".txt"
        (labels_dir / label_name).write_text("\n".join(labels) + "\n" if labels else "")

    # Save batch metadata
    meta = {
        "created": timestamp,
        "batch_size": len(batch_paths),
        "conf_threshold": conf_threshold,
        "images": [p.name for p in batch_paths],
    }
    (batch_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nBatch ready at: {batch_dir}")
    print(f"  {len(batch_paths)} images with pre-labels")
    print("  Import into CVAT:")
    print(f"    1. Create project with labels from {batch_dir}/classes.txt")
    print(f"    2. Upload images from {batch_dir}/images/")
    print(f"    3. Import labels as 'YOLO 1.1' from {batch_dir}/labels/")
    print("    4. Review and correct annotations")
    print("    5. Export as 'YOLO 1.1' for integration")

    return batch_dir


def integrate(
    cvat_export_dir: Path,
    training_data_dir: Path = _TRAINING_DATA_DIR,
) -> int:
    """Integrate corrected CVAT labels into the training dataset.

    Copies images and label files from a CVAT YOLO export into the
    training dataset's train/ directory.

    Args:
        cvat_export_dir: CVAT export directory (should contain images/ and labels/).
        training_data_dir: Target training data directory.

    Returns:
        Number of images integrated.
    """
    cvat_export_dir = Path(cvat_export_dir)
    training_data_dir = Path(training_data_dir)

    # Find images and labels
    cvat_images = cvat_export_dir / "images"
    cvat_labels = cvat_export_dir / "labels"

    # Also check for flat CVAT export (images at root with obj_train_data)
    if not cvat_images.exists():
        cvat_images = cvat_export_dir / "obj_train_data"
    if not cvat_labels.exists():
        cvat_labels = cvat_export_dir / "obj_train_data"

    if not cvat_images.exists():
        print(f"ERROR: No images found in {cvat_export_dir}")
        return 0

    train_images = training_data_dir / "train" / "images"
    train_labels = training_data_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    # Find existing image count for naming
    existing = list(train_images.glob("real_*.jpg")) + list(train_images.glob("real_*.png"))
    next_idx = len(existing)

    count = 0
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    for img_path in sorted(cvat_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Check for corresponding label file
        label_path = cvat_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Copy with sequential naming
        new_name = f"real_{next_idx + count:05d}"
        dest_img = train_images / (new_name + img_path.suffix)
        dest_label = train_labels / (new_name + ".txt")

        shutil.copy2(img_path, dest_img)
        shutil.copy2(label_path, dest_label)
        count += 1

    print(f"Integrated {count} labeled images into {training_data_dir / 'train'}")
    print(f"  Images: {train_images}")
    print(f"  Labels: {train_labels}")
    print(f"  Total real images in training set: {next_idx + count}")

    return count


class _ActiveLearningArgs(argparse.Namespace):
    command: str
    model: str
    input: str
    output: str
    batch_size: int
    conf: float
    cvat_export: str
    training_data: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active learning pipeline for AoE2 detection",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Triage
    triage_parser = subparsers.add_parser("triage", help="Score images by informativeness")
    triage_parser.add_argument("--model", type=str, default=str(_DEFAULT_MODEL))
    triage_parser.add_argument("--input", type=str, default=str(_DEFAULT_RAW_DIR))
    triage_parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT_DIR))

    # Prepare
    prepare_parser = subparsers.add_parser("prepare", help="Prepare CVAT labeling batch")
    prepare_parser.add_argument("--model", type=str, default=str(_DEFAULT_MODEL))
    prepare_parser.add_argument("--input", type=str, default=str(_DEFAULT_RAW_DIR))
    prepare_parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--batch-size", type=int, default=20)
    prepare_parser.add_argument("--conf", type=float, default=0.25)

    # Integrate
    integrate_parser = subparsers.add_parser("integrate", help="Integrate CVAT export")
    integrate_parser.add_argument(
        "--cvat-export", type=str, required=True, help="Path to CVAT YOLO export directory"
    )
    integrate_parser.add_argument("--training-data", type=str, default=str(_TRAINING_DATA_DIR))

    args = parser.parse_args(namespace=_ActiveLearningArgs())

    if args.command == "triage":
        triage(Path(args.model), Path(args.input), Path(args.output))
    elif args.command == "prepare":
        prepare_batch(
            Path(args.model),
            Path(args.input),
            Path(args.output),
            args.batch_size,
            args.conf,
        )
    elif args.command == "integrate":
        integrate(Path(args.cvat_export), Path(args.training_data))


if __name__ == "__main__":
    main()
