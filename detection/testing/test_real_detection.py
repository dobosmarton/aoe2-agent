#!/usr/bin/env python3
"""
Test YOLO Detection on Real Screenshots

This script validates the trained YOLO model on real game screenshots
to verify detection accuracy before integrating with the agent.

Usage:
    python test_real_detection.py                           # Test with v2 model
    python test_real_detection.py --model models/aoe2_yolo_v2.pt
    python test_real_detection.py --images real_screenshots/test --save
    python test_real_detection.py --compare  # Compare v1 vs v2 models

Expected v2 improvements:
    | Metric           | v1 (Synthetic) | v2 (Hybrid) | Target |
    |------------------|----------------|-------------|--------|
    | mAP50 (real)     | ~10%           | 60-70%      | 70%+   |
    | Confidence on TC | 0.34           | 0.7+        | 0.8+   |
    | Villager detect  | 0%             | 50%+        | 70%+   |
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO detection on real game screenshots"
    )
    parser.add_argument(
        "--model", "-m",
        default="detection/inference/models/aoe2_yolo_v2.pt",
        help="Path to YOLO model (default: detection/inference/models/aoe2_yolo_v2.pt)"
    )
    parser.add_argument(
        "--images", "-i",
        default="detection/real_screenshots/test",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save annotated images"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display images with detections"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of images to test"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare v1 vs v2 model performance"
    )
    parser.add_argument(
        "--v1-model",
        default="detection/inference/models/aoe2_yolo26.pt",
        help="Path to v1 model for comparison"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed detection results"
    )

    args = parser.parse_args()

    # Import dependencies
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Install with: pip install Pillow")
        return 1

    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent  # agent/
    model_path = script_dir / args.model
    images_path = script_dir / args.images

    # Check paths
    if not model_path.exists():
        # Try ONNX version
        onnx_path = model_path.with_suffix('.onnx')
        if onnx_path.exists():
            model_path = onnx_path
        else:
            print(f"Error: Model not found: {model_path}")
            print("Train the v2 model first with train_yolo_v2.py")
            return 1

    if not images_path.exists():
        print(f"Error: Images directory not found: {images_path}")
        print("Capture real game screenshots first")
        return 1

    # Get test images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(images_path.glob(ext))

    if not image_files:
        print(f"No images found in: {images_path}")
        return 1

    if args.limit:
        image_files = image_files[:args.limit]

    print("=" * 60)
    print("AoE2 YOLO Detection Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Images: {images_path}")
    print(f"Test images: {len(image_files)}")
    print(f"Confidence: {args.conf}")
    print("=" * 60)

    # Load model
    model = YOLO(str(model_path))

    # Track statistics
    total_detections = 0
    class_counts = {}
    confidence_sum = 0
    images_with_detections = 0

    # Test each image
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")

        results = model(str(img_path), conf=args.conf, save=args.save, verbose=False)

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                num_detections = len(boxes)
                total_detections += num_detections
                images_with_detections += 1

                # Count by class
                for cls_id, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                    class_idx = int(cls_id)
                    class_name = model.names.get(class_idx, f"class_{class_idx}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_sum += conf

                print(f"  Detections: {num_detections}")

                if args.verbose:
                    for box, cls_id, conf in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.cls.cpu().numpy(),
                        boxes.conf.cpu().numpy()
                    ):
                        class_name = model.names.get(int(cls_id), "unknown")
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        print(f"    {class_name}: ({cx:.0f}, {cy:.0f}) conf={conf:.2f}")
            else:
                print("  Detections: 0")
        else:
            print("  Detections: 0")

        # Show image if requested
        if args.show:
            import matplotlib.pyplot as plt
            results[0].show()

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Images tested: {len(image_files)}")
    print(f"Images with detections: {images_with_detections} ({100*images_with_detections/len(image_files):.0f}%)")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.1f}")

    if total_detections > 0:
        avg_conf = confidence_sum / total_detections
        print(f"Average confidence: {avg_conf:.2f}")

    print("\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")

    # Priority class check
    priority_classes = ['villager', 'sheep', 'town_center', 'house', 'gold_mine']
    print("\nPriority classes detected:")
    for cls in priority_classes:
        count = class_counts.get(cls, 0)
        status = "OK" if count > 0 else "MISSING"
        print(f"  {cls}: {count} [{status}]")

    # Comparison mode
    if args.compare:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (v1 vs v2)")
        print("=" * 60)

        v1_path = script_dir / args.v1_model
        if v1_path.exists():
            v1_model = YOLO(str(v1_path))

            v1_detections = 0
            v1_conf_sum = 0

            for img_path in image_files[:5]:  # Quick comparison on first 5
                results = v1_model(str(img_path), conf=args.conf, verbose=False)
                if results and results[0].boxes is not None:
                    v1_detections += len(results[0].boxes)
                    v1_conf_sum += results[0].boxes.conf.sum().item()

            print(f"v1 model ({args.v1_model}):")
            print(f"  Detections (5 images): {v1_detections}")
            if v1_detections > 0:
                print(f"  Avg confidence: {v1_conf_sum/v1_detections:.2f}")

            # Compare with v2 results
            v2_5_detections = sum(1 for _ in image_files[:5])  # Simplified
            print(f"\nImprovement: v1={v1_detections}, v2={total_detections} on {len(image_files)} images")
        else:
            print(f"v1 model not found at {v1_path}")

    if args.save:
        print(f"\nAnnotated images saved to: runs/detect/predict*/")

    return 0


if __name__ == "__main__":
    exit(main())
