"""
Convert YOLO pre-labels to COCO JSON format for reliable CVAT import.

CVAT's YOLO import is unreliable, but COCO 1.0 import works well.
This script reads the prelabeled YOLO .txt files and outputs a
COCO-format instances JSON that can be imported into a CVAT task.

Usage:
    python -m detection.labeling.export_coco [--input DIR] [--output FILE]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

from .class_mapping import load_classes_yaml

_DETECTION_DIR = Path(__file__).parent.parent
_DEFAULT_INPUT = _DETECTION_DIR / "labeling" / "output" / "prelabeled"
_DEFAULT_OUTPUT = _DETECTION_DIR / "labeling" / "output" / "prelabeled" / "instances_default.json"


def yolo_to_coco(
    input_dir: Path = _DEFAULT_INPUT,
    output_path: Path = _DEFAULT_OUTPUT,
) -> Path:
    """Convert YOLO pre-labels to COCO JSON.

    Args:
        input_dir: Directory with images/, labels/, and classes.txt.
        output_path: Where to write the COCO JSON file.

    Returns:
        Path to the output JSON file.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        print(f"ERROR: images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: labels directory not found: {labels_dir}")
        sys.exit(1)

    # Load class names from classes.yaml (v2 schema)
    v2_classes = load_classes_yaml()

    # Build COCO categories (1-indexed as per COCO convention)
    categories = []
    for class_id in sorted(v2_classes.keys()):
        categories.append({
            "id": class_id + 1,  # COCO is 1-indexed
            "name": v2_classes[class_id],
            "supercategory": "",
        })

    # Find all images (follow symlinks)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    print(f"Processing {len(image_paths)} images...")

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_idx, img_path in enumerate(image_paths):
        # Resolve symlinks to get actual file
        real_path = img_path.resolve() if img_path.is_symlink() else img_path

        # Get image dimensions
        with Image.open(real_path) as img:
            img_w, img_h = img.size

        image_id = img_idx + 1
        # Use the original filename (what CVAT sees)
        file_name = img_path.name

        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
        })

        # Read corresponding YOLO label
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        label_text = label_path.read_text().strip()
        if not label_text:
            continue

        for line in label_text.split("\n"):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Convert normalized YOLO to COCO pixel coordinates
            # COCO bbox = [x_top_left, y_top_left, width, height]
            bbox_w = w_norm * img_w
            bbox_h = h_norm * img_h
            bbox_x = (x_center * img_w) - (bbox_w / 2)
            bbox_y = (y_center * img_h) - (bbox_h / 2)

            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # COCO is 1-indexed
                "bbox": [
                    round(bbox_x, 2),
                    round(bbox_y, 2),
                    round(bbox_w, 2),
                    round(bbox_h, 2),
                ],
                "area": round(bbox_w * bbox_h, 2),
                "segmentation": [],
                "iscrowd": 0,
            })
            ann_id += 1

    # Assemble COCO JSON
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2) + "\n")

    print(f"\nCOCO JSON written to: {output_path}")
    print(f"  {len(coco_images)} images")
    print(f"  {len(coco_annotations)} annotations")
    print(f"  {len(categories)} categories")
    print(f"\nImport into CVAT:")
    print(f"  1. Open your CVAT task")
    print(f"  2. Actions → Upload annotations")
    print(f"  3. Select format: 'COCO 1.0'")
    print(f"  4. Upload: {output_path.name}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO pre-labels to COCO JSON for CVAT import",
    )
    parser.add_argument(
        "--input", type=str, default=str(_DEFAULT_INPUT),
        help="Prelabeled directory with images/ and labels/",
    )
    parser.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output COCO JSON path",
    )
    args = parser.parse_args()

    yolo_to_coco(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
