"""
Prepare hybrid training dataset by merging CVAT-labeled real screenshots
with synthetic training data.

Takes a CVAT export (COCO 1.0 or YOLO 1.1) and combines it with the existing
synthetic dataset to create a unified training set for fine-tuning.

Usage:
    # Merge CVAT export into training dataset (auto-detects COCO or YOLO format)
    python -m detection.labeling.prepare_training --cvat-export /path/to/cvat/export

    # Dry run (show what would happen without copying)
    python -m detection.labeling.prepare_training --cvat-export /path/to/export --dry-run

    # Custom output directory
    python -m detection.labeling.prepare_training --cvat-export /path/to/export --output /path/to/output

CVAT Export Instructions:
    1. In CVAT, select your task/project
    2. Click "Export task dataset"
    3. Choose "COCO 1.0" format (supports both rectangle AND polygon annotations)
       - Alternative: "YOLO 1.1" (rectangles only, polygons will be lost!)
    4. Download and unzip the export
    5. Pass the unzipped directory to this script
"""

import argparse
import json
import random
import shutil
import sys
import yaml
from pathlib import Path

from .class_mapping import build_v1_to_v2_mapping, convert_label_file, load_classes_yaml, load_dataset_yaml

# Paths
_DETECTION_DIR = Path(__file__).parent.parent
_SYNTHETIC_DIR = _DETECTION_DIR / "training_data"
_DEFAULT_OUTPUT = _DETECTION_DIR / "training_data_v2"


def detect_export_format(cvat_dir: Path) -> str:
    """Detect whether a CVAT export is COCO or YOLO format.

    Returns:
        "coco" or "yolo".
    """
    cvat_dir = Path(cvat_dir)

    # COCO: look for instances_default.json or annotations/*.json
    if (cvat_dir / "annotations").exists():
        json_files = list((cvat_dir / "annotations").glob("*.json"))
        if json_files:
            return "coco"

    # Also check root for a COCO JSON
    for f in cvat_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "annotations" in data and "images" in data:
                return "coco"
        except (json.JSONDecodeError, KeyError):
            continue

    return "yolo"


def convert_coco_to_yolo_labels(
    cvat_dir: Path,
    output_labels_dir: Path,
    v2_classes: dict[int, str],
) -> list[tuple[str, Path, int]]:
    """Convert COCO format CVAT export to YOLO label files.

    Handles both bbox and polygon annotations by computing the bounding
    box from polygon vertices when needed.

    Args:
        cvat_dir: CVAT COCO export directory.
        output_labels_dir: Where to write .txt YOLO label files.
        v2_classes: Target class scheme {id: name}.

    Returns:
        List of (image_filename, label_path, n_labels) tuples for images
        that have at least one annotation.
    """
    # Find the COCO JSON file
    coco_json = None
    annotations_dir = cvat_dir / "annotations"
    if annotations_dir.exists():
        json_files = list(annotations_dir.glob("*.json"))
        if json_files:
            coco_json = json_files[0]  # typically instances_default.json

    if coco_json is None:
        # Check root
        for f in cvat_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "annotations" in data and "images" in data:
                    coco_json = f
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if coco_json is None:
        raise FileNotFoundError(f"No COCO JSON found in {cvat_dir}")

    print(f"  Reading COCO annotations from: {coco_json.name}")
    coco_data = json.loads(coco_json.read_text(encoding="utf-8"))

    # Build COCO category_id -> v2 class_id mapping via name matching
    v2_name_to_id = {name: cid for cid, name in v2_classes.items()}
    coco_cat_to_v2 = {}
    for cat in coco_data.get("categories", []):
        cat_name = cat["name"]
        if cat_name in v2_name_to_id:
            coco_cat_to_v2[cat["id"]] = v2_name_to_id[cat_name]
        else:
            print(f"  WARNING: COCO category '{cat_name}' not found in classes.yaml, skipping")

    print(f"  Mapped {len(coco_cat_to_v2)}/{len(coco_data.get('categories', []))} COCO categories to v2 IDs")

    # Build image_id -> image info
    images_by_id = {}
    for img in coco_data.get("images", []):
        images_by_id[img["id"]] = img

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Convert each image's annotations to YOLO format
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for img_id, img_info in images_by_id.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]
        stem = Path(file_name).stem

        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        yolo_lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in coco_cat_to_v2:
                continue
            v2_id = coco_cat_to_v2[cat_id]

            # Get bounding box - prefer bbox field, fall back to polygon
            bbox = ann.get("bbox")
            if bbox and bbox[2] > 0 and bbox[3] > 0:
                # COCO bbox: [x_top_left, y_top_left, width, height]
                x, y, w, h = bbox
            elif ann.get("segmentation"):
                # Compute bbox from polygon points
                seg = ann["segmentation"]
                if isinstance(seg, list) and len(seg) > 0:
                    # Flatten all polygon points
                    points = seg[0] if isinstance(seg[0], list) else seg
                    xs = points[0::2]
                    ys = points[1::2]
                    if not xs or not ys:
                        continue
                    x = min(xs)
                    y = min(ys)
                    w = max(xs) - x
                    h = max(ys) - y
                else:
                    continue
            else:
                continue

            if w <= 0 or h <= 0:
                continue

            # Convert to YOLO normalized format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_lines.append(f"{v2_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if yolo_lines:
            label_path = output_labels_dir / (stem + ".txt")
            label_path.write_text("\n".join(yolo_lines) + "\n")
            results.append((file_name, label_path, len(yolo_lines)))

    print(f"  Converted {len(results)} images with {sum(r[2] for r in results)} total annotations")
    return results


def find_cvat_labels(cvat_dir: Path) -> Path:
    """Find the directory containing YOLO .txt label files in a CVAT export.

    CVAT YOLO exports can have different structures:
      - obj_train_data/ with .txt labels (and optionally images)
      - labels/ directory
      - train/labels/ directory
      - .txt files at root level
    """
    cvat_dir = Path(cvat_dir)

    # Check for obj_train_data (standard CVAT YOLO export, case-insensitive)
    for name in ("obj_train_data", "obj_Train_data"):
        obj_dir = cvat_dir / name
        if obj_dir.exists() and list(obj_dir.glob("*.txt")):
            return obj_dir

    # Check for labels/ directory
    if (cvat_dir / "labels").exists():
        return cvat_dir / "labels"

    # Check for train/labels/
    if (cvat_dir / "train" / "labels").exists():
        return cvat_dir / "train" / "labels"

    # Fallback: .txt files at root
    if list(cvat_dir.glob("*.txt")):
        return cvat_dir

    raise FileNotFoundError(
        f"Could not find YOLO label files in {cvat_dir}. "
        f"Expected obj_train_data/ or labels/ with .txt files."
    )


def count_labels(label_path: Path) -> int:
    """Count number of bounding boxes in a YOLO label file."""
    if not label_path.exists():
        return 0
    text = label_path.read_text().strip()
    if not text:
        return 0
    return len(text.split("\n"))


def prepare_training(
    cvat_export_dir: str | Path,
    output_dir: str | Path = _DEFAULT_OUTPUT,
    synthetic_dir: str | Path = _SYNTHETIC_DIR,
    images_dir: str | Path | None = None,
    val_split: float = 0.15,
    include_synthetic: bool = True,
    dry_run: bool = False,
) -> dict:
    """Merge CVAT real labels with synthetic data into a training dataset.

    Supports labels-only CVAT exports by matching label filenames against
    images in a local directory (default: real_screenshots/raw/).

    Args:
        cvat_export_dir: Path to unzipped CVAT YOLO export.
        output_dir: Where to write the merged dataset.
        synthetic_dir: Path to existing synthetic training data.
        images_dir: Local directory with source images. If None, uses
                    real_screenshots/raw/ as default.
        val_split: Fraction of real images to use for validation.
        include_synthetic: Whether to include synthetic data in the merge.
        dry_run: If True, just report what would happen.

    Returns:
        Summary dict with counts.
    """
    cvat_export_dir = Path(cvat_export_dir)
    output_dir = Path(output_dir)
    synthetic_dir = Path(synthetic_dir)
    if images_dir is None:
        images_dir = _DETECTION_DIR / "real_screenshots" / "raw"
    images_dir = Path(images_dir)

    # Build index of local images by stem name for fast lookup
    print(f"Scanning images directory: {images_dir}")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    local_images = {}
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            local_images[img_path.stem] = img_path
    print(f"  Found {len(local_images)} images locally")

    # Auto-detect export format and extract labels
    print(f"\nScanning CVAT export: {cvat_export_dir}")
    export_format = detect_export_format(cvat_export_dir)
    print(f"  Detected format: {export_format.upper()}")

    real_pairs = []
    missing_images = []

    if export_format == "coco":
        # Convert COCO annotations to YOLO labels in a temp directory
        v2_classes = load_classes_yaml()
        _tmp_labels = output_dir / "_tmp_coco_labels"
        coco_results = convert_coco_to_yolo_labels(cvat_export_dir, _tmp_labels, v2_classes)

        for file_name, label_path, n_labels in coco_results:
            stem = Path(file_name).stem
            if stem in local_images:
                real_pairs.append((local_images[stem], label_path, n_labels))
            else:
                missing_images.append(stem)
    else:
        # YOLO format - find label files directly
        try:
            cvat_labels_dir = find_cvat_labels(cvat_export_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        for label_path in sorted(cvat_labels_dir.glob("*.txt")):
            if label_path.stem in ("obj", "train", "classes"):
                continue
            n_labels = count_labels(label_path)
            if n_labels == 0:
                continue
            stem = label_path.stem
            if stem in local_images:
                real_pairs.append((local_images[stem], label_path, n_labels))
            else:
                missing_images.append(stem)

    if missing_images:
        print(f"  WARNING: {len(missing_images)} labels have no matching local image:")
        for name in missing_images[:5]:
            print(f"    {name}")
        if len(missing_images) > 5:
            print(f"    ... and {len(missing_images) - 5} more")

    print(f"Found {len(real_pairs)} labeled real images")

    # Split real data into train/val
    random.seed(42)
    shuffled = list(real_pairs)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    real_val = shuffled[:n_val]
    real_train = shuffled[n_val:]

    print(f"  Train: {len(real_train)}, Val: {len(real_val)}")

    # Count synthetic data
    synth_train_count = 0
    synth_val_count = 0
    if include_synthetic and synthetic_dir.exists():
        synth_train_images = synthetic_dir / "train" / "images"
        synth_val_images = synthetic_dir / "val" / "images"
        if synth_train_images.exists():
            synth_train_count = len(list(synth_train_images.glob("*.jpg")))
        if synth_val_images.exists():
            synth_val_count = len(list(synth_val_images.glob("*.jpg")))
        print(f"Synthetic data: {synth_train_count} train, {synth_val_count} val")

    # Summary
    total_train = len(real_train) + synth_train_count
    total_val = len(real_val) + synth_val_count

    summary = {
        "real_train": len(real_train),
        "real_val": len(real_val),
        "synthetic_train": synth_train_count,
        "synthetic_val": synth_val_count,
        "total_train": total_train,
        "total_val": total_val,
    }

    print(f"\nMerged dataset will contain:")
    print(f"  Train: {total_train} ({len(real_train)} real + {synth_train_count} synthetic)")
    print(f"  Val:   {total_val} ({len(real_val)} real + {synth_val_count} synthetic)")

    if dry_run:
        print("\n[DRY RUN] No files copied.")
        return summary

    # Create output directory structure
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    val_images = output_dir / "val" / "images"
    val_labels = output_dir / "val" / "labels"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy synthetic data (if including) with class ID remapping
    if include_synthetic and synthetic_dir.exists():
        print("\nCopying synthetic data (remapping v1 -> v2 class IDs)...")
        v1_to_v2 = build_v1_to_v2_mapping()
        _copy_dir_contents(synthetic_dir / "train" / "images", train_images)
        _remap_labels(synthetic_dir / "train" / "labels", train_labels, v1_to_v2)
        _copy_dir_contents(synthetic_dir / "val" / "images", val_images)
        _remap_labels(synthetic_dir / "val" / "labels", val_labels, v1_to_v2)

    # Copy real training data
    print("Copying real training data...")
    for img_path, label_path, _ in real_train:
        new_name = f"real_{img_path.stem}"
        _copy_pair(img_path, label_path, train_images, train_labels, new_name)

    # Copy real validation data
    print("Copying real validation data...")
    for img_path, label_path, _ in real_val:
        new_name = f"real_{img_path.stem}"
        _copy_pair(img_path, label_path, val_images, val_labels, new_name)

    # Generate dataset.yaml
    classes = load_classes_yaml()
    dataset_yaml = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "names": {int(k): v for k, v in classes.items()},
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset written to: {output_dir}")
    print(f"  dataset.yaml: {yaml_path}")
    print(f"  Train: {total_train} images")
    print(f"  Val:   {total_val} images")
    print(f"  Classes: {len(classes)}")

    # Clean up temp COCO labels if created
    _tmp_labels = output_dir / "_tmp_coco_labels"
    if _tmp_labels.exists():
        shutil.rmtree(_tmp_labels)

    # Save summary
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nNext step: Upload to Lambda Labs and train:")
    print(f"  tar -czf training_data_v2.tar.gz -C {output_dir.parent} {output_dir.name}")
    print(f"  scp training_data_v2.tar.gz ubuntu@<LAMBDA_IP>:/home/ubuntu/")

    return summary


def _copy_dir_contents(src: Path, dst: Path) -> int:
    """Copy all files from src to dst directory."""
    if not src.exists():
        return 0
    count = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def _remap_labels(src: Path, dst: Path, mapping: dict[int, int]) -> int:
    """Copy label files from src to dst, remapping class IDs."""
    if not src.exists():
        return 0
    count = 0
    for f in src.iterdir():
        if f.is_file() and f.suffix == ".txt":
            convert_label_file(f, dst / f.name, mapping, skip_unmapped=True)
            count += 1
    return count


def _copy_pair(
    img_path: Path,
    label_path: Path,
    images_dir: Path,
    labels_dir: Path,
    new_name: str,
) -> None:
    """Copy an image-label pair with a new base name."""
    img_dest = images_dir / (new_name + img_path.suffix.lower())
    label_dest = labels_dir / (new_name + ".txt")
    shutil.copy2(img_path, img_dest)
    shutil.copy2(label_path, label_dest)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hybrid training dataset from CVAT export + synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cvat-export", type=str, required=True,
        help="Path to unzipped CVAT export directory (COCO 1.0 or YOLO 1.1)",
    )
    parser.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT),
        help=f"Output directory (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="Directory with source images (default: real_screenshots/raw/)",
    )
    parser.add_argument(
        "--synthetic", type=str, default=str(_SYNTHETIC_DIR),
        help="Path to existing synthetic training data",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of real images for validation (default: 0.15)",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Exclude synthetic data (real images only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without copying files",
    )
    args = parser.parse_args()

    prepare_training(
        cvat_export_dir=args.cvat_export,
        output_dir=args.output,
        synthetic_dir=args.synthetic,
        images_dir=args.images_dir,
        val_split=args.val_split,
        include_synthetic=not args.no_synthetic,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
