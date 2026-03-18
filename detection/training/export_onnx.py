"""Export YOLO model to ONNX format for batch inference.

Usage:
    python -m detection.training.export_onnx
    python -m detection.training.export_onnx --model detection/inference/models/aoe2_yolo_v5.pt
"""

import argparse
from pathlib import Path


def export_to_onnx(model_path: str, imgsz: int = 640) -> str:
    """Export a YOLO .pt model to ONNX format with dynamic batch support.

    Args:
        model_path: Path to the .pt model file
        imgsz: Input image size (must match training size)

    Returns:
        Path to the exported .onnx file
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"Exporting {model_path} to ONNX (imgsz={imgsz}, dynamic batch)...")

    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=True,  # Enable dynamic batch size for batched SAHI
    )

    print(f"Exported to: {export_path}")
    return str(export_path)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent.parent / "inference" / "models" / "aoe2_yolo_v5.pt"),
        help="Path to .pt model file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    export_to_onnx(args.model, args.imgsz)


if __name__ == "__main__":
    main()
