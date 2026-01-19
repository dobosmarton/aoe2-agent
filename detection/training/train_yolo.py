#!/usr/bin/env python3
"""
YOLO v2 Training Script for AoE2 Detection

This script trains an improved YOLO model using the v2 hybrid dataset
that combines real annotated screenshots with improved synthetic data.

Usage:
    python train_yolo_v2.py                          # Train with defaults
    python train_yolo_v2.py --epochs 200             # Custom epochs
    python train_yolo_v2.py --model yolo11s.pt       # Use larger model
    python train_yolo_v2.py --resume                 # Resume training

Lambda Labs Training:
    1. Upload dataset: scp -r detection/training_data_v2 ubuntu@<IP>:/home/ubuntu/
    2. Upload script:  scp train_yolo_v2.py ubuntu@<IP>:/home/ubuntu/
    3. SSH and run:    python train_yolo_v2.py
    4. Download:       scp ubuntu@<IP>:/home/ubuntu/runs/aoe2_yolo_v2/weights/best.pt ./

Estimated training time: ~2 hours on A100 for 150 epochs
Estimated cost: ~$2.60 on Lambda Labs
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO v2 model for AoE2 detection"
    )
    parser.add_argument(
        "--data", "-d",
        default="detection/training_data_v2/dataset.yaml",
        help="Path to dataset.yaml (default: detection/training_data_v2/dataset.yaml)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="Base model to use (default: yolo11n.pt for YOLO11 nano)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16, adjust based on GPU memory)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Project directory for outputs (default: runs)"
    )
    parser.add_argument(
        "--name",
        default="aoe2_yolo_v2",
        help="Run name (default: aoe2_yolo_v2)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export to ONNX after training"
    )

    args = parser.parse_args()

    # Import ultralytics (may not be installed on all systems)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return 1

    # Resolve dataset path
    script_dir = Path(__file__).parent.parent  # agent/
    data_path = script_dir / args.data
    if not data_path.exists():
        # Try relative to current directory
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Dataset not found: {args.data}")
            print("Generate the dataset first with generate_training_data.py")
            return 1

    print("=" * 60)
    print("AoE2 YOLO v2 Training")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: cuda:{args.device}")
    print("=" * 60)

    # Load model
    if args.resume:
        checkpoint_path = Path(args.project) / args.name / "weights" / "last.pt"
        if checkpoint_path.exists():
            print(f"Resuming from: {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # Train with optimized hyperparameters for AoE2
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,

        # Augmentation settings optimized for isometric game graphics
        hsv_h=0.015,          # Hue augmentation (slight)
        hsv_s=0.7,            # Saturation augmentation
        hsv_v=0.4,            # Value/brightness augmentation
        degrees=10,           # Rotation (units can face different directions)
        translate=0.1,        # Translation
        scale=0.5,            # Scale augmentation (important for zoom levels)
        flipud=0.0,           # No vertical flip (isometric view)
        fliplr=0.5,           # Horizontal flip OK
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.1,            # MixUp augmentation

        # Output settings
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Get best model path
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best model: {best_model_path}")

    # Export to ONNX if requested
    if args.export_onnx and best_model_path.exists():
        print("\nExporting to ONNX...")
        best_model = YOLO(str(best_model_path))
        onnx_path = best_model.export(format='onnx', imgsz=args.imgsz, simplify=True)
        print(f"ONNX model: {onnx_path}")

        # Copy to detection/models
        models_dir = script_dir / "detection" / "models"
        models_dir.mkdir(exist_ok=True)

        import shutil
        dest_pt = models_dir / "aoe2_yolo_v2.pt"
        dest_onnx = models_dir / "aoe2_yolo_v2.onnx"

        shutil.copy(best_model_path, dest_pt)
        shutil.copy(onnx_path, dest_onnx)

        print(f"\nModels copied to:")
        print(f"  {dest_pt}")
        print(f"  {dest_onnx}")

    return 0


if __name__ == "__main__":
    exit(main())
