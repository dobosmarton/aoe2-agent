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
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

from .class_mapping import (
    build_v1_to_v2_mapping,
    load_classes_yaml,
    load_dataset_yaml,
    write_classes_txt,
)

# Paths
_DETECTION_DIR = Path(__file__).parent.parent
_DEFAULT_MODEL = _DETECTION_DIR / "inference" / "models" / "aoe2_yolo_v2.pt"
_DEFAULT_RAW_DIR = _DETECTION_DIR / "real_screenshots" / "raw"
_DEFAULT_OUTPUT_DIR = _DETECTION_DIR / "labeling" / "output" / "active_learning"
_TRAINING_DATA_DIR = _DETECTION_DIR / "training_data"


def triage(
    model_path: Path = _DEFAULT_MODEL,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    conf_low: float = 0.15,
    conf_high: float = 0.7,
) -> list[dict]:
    """Score all images by informativeness for active learning.

    Images with many uncertain or missing detections are most valuable
    to label because they teach the model the most.

    Scoring:
        - Detections with conf < conf_low: +3 points (model is clueless)
        - Detections with conf_low <= conf < conf_high: +2 (model is uncertain)
        - No detections at all: +15 points (image may contain novel content)
        - Fewer total detections than expected: +5 (probably missing objects)

    Returns:
        Sorted list of {path, score, n_detections, n_uncertain, n_low} dicts.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is required.")
        sys.exit(1)

    model = YOLO(str(model_path))
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted([
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    print(f"Triaging {len(images)} images...")

    scored = []
    for i, img_path in enumerate(images):
        results = model(str(img_path), conf=0.05, verbose=False)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            scored.append({
                "path": str(img_path),
                "name": img_path.name,
                "score": 15,
                "n_detections": 0,
                "n_uncertain": 0,
                "n_low": 0,
                "reason": "no_detections",
            })
            continue

        confs = results[0].boxes.conf.cpu().numpy()
        n_total = len(confs)
        n_low = int(sum(1 for c in confs if c < conf_low))
        n_uncertain = int(sum(1 for c in confs if conf_low <= c < conf_high))
        n_high = int(sum(1 for c in confs if c >= conf_high))

        # Score: uncertain and low-confidence detections are most informative
        score = n_low * 3 + n_uncertain * 2

        # Bonus if very few detections (probably missing many objects)
        if n_total < 5:
            score += 5

        scored.append({
            "path": str(img_path),
            "name": img_path.name,
            "score": score,
            "n_detections": n_total,
            "n_uncertain": n_uncertain,
            "n_low": n_low,
            "n_high": n_high,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(images)}] processed")

    # Sort by score descending (most informative first)
    scored.sort(key=lambda x: -x["score"])

    # Save triage results
    output_dir.mkdir(parents=True, exist_ok=True)
    triage_path = output_dir / "triage_results.json"
    triage_path.write_text(json.dumps(scored, indent=2) + "\n")

    print(f"\nTriage complete. Results saved to {triage_path}")
    print(f"\nTop 10 most informative images:")
    for item in scored[:10]:
        print(
            f"  Score {item['score']:3d} | {item['n_detections']:3d} det "
            f"({item['n_uncertain']} uncertain, {item['n_low']} low) | {item['name']}"
        )

    print(f"\nBottom 5 (least informative):")
    for item in scored[-5:]:
        print(
            f"  Score {item['score']:3d} | {item['n_detections']:3d} det "
            f"({item.get('n_high', 0)} high-conf) | {item['name']}"
        )

    return scored


def prepare_batch(
    model_path: Path = _DEFAULT_MODEL,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    batch_size: int = 20,
    conf_threshold: float = 0.25,
    schema: str = "v2",
) -> Path:
    """Prepare the next batch of most-informative images for CVAT labeling.

    Runs triage if not already done, then creates a CVAT-importable
    directory with images and pre-labels.

    Returns:
        Path to the batch directory.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is required.")
        sys.exit(1)

    output_dir = Path(output_dir)

    # Check for existing triage results
    triage_path = output_dir / "triage_results.json"
    if triage_path.exists():
        print("Loading existing triage results...")
        scored = json.loads(triage_path.read_text())
    else:
        print("No triage results found, running triage...")
        scored = triage(model_path, raw_dir, output_dir)

    # Select top-N images
    batch = scored[:batch_size]
    print(f"\nPreparing batch of {len(batch)} images...")

    # Create batch directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    batch_dir = output_dir / f"batch_{timestamp}"
    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load model for pre-labeling
    model = YOLO(str(model_path))

    # Build class mapping
    v1_classes = load_dataset_yaml()
    v2_classes = load_classes_yaml()
    if schema == "v2":
        id_mapping = build_v1_to_v2_mapping(v1_classes, v2_classes)
        output_classes = v2_classes
    else:
        id_mapping = {i: i for i in v1_classes}
        output_classes = v1_classes

    # Write classes.txt
    write_classes_txt(batch_dir / "classes.txt", schema)

    # Process each image
    for item in batch:
        img_path = Path(item["path"])
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

            for box, cls_id, conf in zip(
                results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
            ):
                v1_id = int(cls_id.item())
                if v1_id not in id_mapping:
                    continue
                mapped_id = id_mapping[v1_id]

                x1, y1, x2, y2 = box.tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                labels.append(f"{mapped_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write label file
        label_name = img_path.stem + ".txt"
        (labels_dir / label_name).write_text("\n".join(labels) + "\n" if labels else "")

    # Save batch metadata
    meta = {
        "created": timestamp,
        "batch_size": len(batch),
        "schema": schema,
        "conf_threshold": conf_threshold,
        "images": [item["name"] for item in batch],
    }
    (batch_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nBatch ready at: {batch_dir}")
    print(f"  {len(batch)} images with pre-labels")
    print(f"  Import into CVAT:")
    print(f"    1. Create project with labels from {batch_dir}/classes.txt")
    print(f"    2. Upload images from {batch_dir}/images/")
    print(f"    3. Import labels as 'YOLO 1.1' from {batch_dir}/labels/")
    print(f"    4. Review and correct annotations")
    print(f"    5. Export as 'YOLO 1.1' for integration")

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


def main():
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
    prepare_parser.add_argument("--schema", choices=["v1", "v2"], default="v2")

    # Integrate
    integrate_parser = subparsers.add_parser("integrate", help="Integrate CVAT export")
    integrate_parser.add_argument("--cvat-export", type=str, required=True,
                                  help="Path to CVAT YOLO export directory")
    integrate_parser.add_argument("--training-data", type=str,
                                  default=str(_TRAINING_DATA_DIR))

    args = parser.parse_args()

    if args.command == "triage":
        triage(Path(args.model), Path(args.input), Path(args.output))
    elif args.command == "prepare":
        prepare_batch(
            Path(args.model), Path(args.input), Path(args.output),
            args.batch_size, args.conf, args.schema,
        )
    elif args.command == "integrate":
        integrate(Path(args.cvat_export), Path(args.training_data))


if __name__ == "__main__":
    main()
