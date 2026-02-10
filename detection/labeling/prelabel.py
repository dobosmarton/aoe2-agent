"""
YOLO model-assisted pre-labeling for AoE2 screenshots.

Runs the existing v1 YOLO model on raw screenshots and exports
predictions as YOLO-format annotations for import into CVAT.

Usage:
    python -m detection.labeling.prelabel [options]

    # Pre-label all raw screenshots
    python -m detection.labeling.prelabel

    # Custom confidence threshold (lower = more boxes to review)
    python -m detection.labeling.prelabel --conf 0.15

    # Use v1 class IDs (skip mapping to classes.yaml)
    python -m detection.labeling.prelabel --schema v1

CVAT Import Workflow:
    1. Create a CVAT project with labels from output/prelabeled/classes.txt
    2. Create a task, upload images from output/prelabeled/images/
    3. Import annotations: upload labels/ folder as "YOLO 1.1" format
    4. Review and correct pre-filled annotations
    5. Export corrected annotations as "YOLO 1.1" for training
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
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
_DEFAULT_OUTPUT_DIR = _DETECTION_DIR / "labeling" / "output" / "prelabeled"

# Colors for preview visualization (one per category)
_CATEGORY_COLORS = {
    "resource": (34, 139, 34),      # green
    "building": (65, 105, 225),     # blue
    "unit": (220, 20, 60),          # red
    "animal": (255, 165, 0),        # orange
    "defense": (148, 0, 211),       # purple
    "special": (255, 215, 0),       # gold
}

# Map class names to categories for coloring
_CLASS_CATEGORIES = {}
_RESOURCE_CLASSES = {"tree", "gold_mine", "stone_mine", "berry_bush", "relic"}
_ANIMAL_CLASSES = {"sheep", "deer", "boar", "wolf"}
_BUILDING_CLASSES = {
    "town_center", "house", "lumber_camp", "mining_camp", "mill", "market",
    "dock", "farm", "barracks", "archery_range", "stable", "blacksmith",
    "siege_workshop", "monastery", "castle", "university",
}
_DEFENSE_CLASSES = {"gate", "wall", "tower", "wonder", "krepost"}


def _get_color(class_name: str) -> tuple[int, int, int]:
    """Get visualization color for a class name."""
    if class_name in _RESOURCE_CLASSES:
        return _CATEGORY_COLORS["resource"]
    elif class_name in _ANIMAL_CLASSES:
        return _CATEGORY_COLORS["animal"]
    elif class_name in _BUILDING_CLASSES:
        return _CATEGORY_COLORS["building"]
    elif class_name in _DEFENSE_CLASSES:
        return _CATEGORY_COLORS["defense"]
    else:
        return _CATEGORY_COLORS["unit"]


def prelabel(
    model_path: str | Path = _DEFAULT_MODEL,
    screenshots_dir: str | Path = _DEFAULT_RAW_DIR,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    conf_threshold: float = 0.25,
    schema: str = "v2",
    save_preview: bool = True,
) -> dict:
    """Run YOLO model on screenshots and export YOLO-format labels.

    Args:
        model_path: Path to YOLO .pt model.
        screenshots_dir: Directory with raw screenshots.
        output_dir: Where to write labels, images, previews.
        conf_threshold: Minimum confidence for detections.
        schema: "v1" to keep model IDs, "v2" to map to classes.yaml.
        save_preview: Whether to save annotated preview images.

    Returns:
        Summary dict with per-class detection counts.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is required. Install with: pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_path)
    screenshots_dir = Path(screenshots_dir)
    output_dir = Path(output_dir)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    if not screenshots_dir.exists():
        print(f"ERROR: Screenshots directory not found: {screenshots_dir}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Build class mapping
    v2_classes = load_classes_yaml()

    if schema == "v2":
        # Check if the model is already v2 (59 classes) or v1 (46 classes)
        model_nc = getattr(model.model, "nc", None) or len(getattr(model, "names", {}))
        if model_nc == len(v2_classes):
            # v2 model outputs v2 IDs natively - identity mapping
            id_mapping = {i: i for i in v2_classes}
            print(f"Using v2 model ({model_nc} classes), no remapping needed")
        else:
            # v1 model needs remapping
            v1_classes = load_dataset_yaml()
            id_mapping = build_v1_to_v2_mapping(v1_classes, v2_classes)
            print(f"Using v1 model ({model_nc} classes), mapped {len(id_mapping)} to v2 schema")
        output_classes = v2_classes
    else:
        v1_classes = load_dataset_yaml()
        id_mapping = {i: i for i in v1_classes}  # identity
        output_classes = v1_classes
        print(f"Using v1 schema ({len(v1_classes)} classes)")

    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Write classes.txt for CVAT
    write_classes_txt(output_dir / "classes.txt", schema)
    print(f"Wrote classes.txt ({schema} schema)")

    # Find all screenshots
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted([
        p for p in screenshots_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])
    print(f"Found {len(images)} screenshots in {screenshots_dir}")

    # Process each image
    summary: dict[str, int] = {}
    total_detections = 0

    for i, img_path in enumerate(images):
        # Run detection
        results = model(str(img_path), conf=conf_threshold, verbose=False)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            # No detections - write empty label file
            label_name = img_path.stem + ".txt"
            (labels_dir / label_name).write_text("")
            _link_image(img_path, images_dir)
            print(f"  [{i+1}/{len(images)}] {img_path.name}: 0 detections")
            continue

        boxes = results[0].boxes
        img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

        # Convert to YOLO labels
        labels = []
        detections_for_preview = []

        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            v1_id = int(cls_id.item())

            # Map class ID
            if v1_id not in id_mapping:
                continue
            mapped_id = id_mapping[v1_id]
            class_name = output_classes.get(mapped_id, f"class_{mapped_id}")

            # Convert to YOLO normalized format
            x1, y1, x2, y2 = box.tolist()
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            labels.append(f"{mapped_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # Track for preview and summary
            detections_for_preview.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": float(conf.item()),
            })

            summary[class_name] = summary.get(class_name, 0) + 1
            total_detections += 1

        # Write label file
        label_name = img_path.stem + ".txt"
        (labels_dir / label_name).write_text("\n".join(labels) + "\n" if labels else "")

        # Link/copy image
        _link_image(img_path, images_dir)

        # Save preview
        if save_preview and detections_for_preview:
            _save_preview(img_path, detections_for_preview, preview_dir)

        print(f"  [{i+1}/{len(images)}] {img_path.name}: {len(labels)} detections")

    # Write summary
    summary_data = {
        "total_images": len(images),
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / max(len(images), 1), 1),
        "confidence_threshold": conf_threshold,
        "schema": schema,
        "per_class": dict(sorted(summary.items(), key=lambda x: -x[1])),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2) + "\n")

    print(f"\nDone! {total_detections} detections across {len(images)} images")
    print(f"Output: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"\nTop classes:")
    for name, count in sorted(summary.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count}")

    return summary_data


def _link_image(src: Path, dest_dir: Path) -> None:
    """Create a symlink or copy an image to the output directory."""
    dest = dest_dir / src.name
    if dest.exists():
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def _save_preview(
    img_path: Path,
    detections: list[dict],
    preview_dir: Path,
) -> None:
    """Save an annotated preview image with bounding boxes."""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        color = _get_color(class_name)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        label = f"{class_name} {conf:.0%}"
        text_bbox = draw.textbbox((x1, y1 - 14), label)
        draw.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1],
            fill=color,
        )
        draw.text((x1, y1 - 14), label, fill=(255, 255, 255))

    # Convert RGBA to RGB for JPEG compatibility
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Save as JPEG to save space
    preview_path = preview_dir / (img_path.stem + ".jpg")
    img.save(preview_path, "JPEG", quality=85)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-label AoE2 screenshots using YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default=str(_DEFAULT_MODEL),
        help="Path to YOLO model (default: inference/models/aoe2_yolo26.pt)",
    )
    parser.add_argument(
        "--input", type=str, default=str(_DEFAULT_RAW_DIR),
        help="Directory with raw screenshots (default: real_screenshots/raw/)",
    )
    parser.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory (default: labeling/output/prelabeled/)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25, lower catches more)",
    )
    parser.add_argument(
        "--schema", choices=["v1", "v2"], default="v2",
        help="Class ID schema: v1 (46-class model) or v2 (55-class target)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Skip generating preview images",
    )
    args = parser.parse_args()

    prelabel(
        model_path=args.model,
        screenshots_dir=args.input,
        output_dir=args.output,
        conf_threshold=args.conf,
        schema=args.schema,
        save_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
