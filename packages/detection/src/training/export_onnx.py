"""Export YOLO model to ONNX format for batch inference.

Usage:
    python -m detection.training.export_onnx
    python -m detection.training.export_onnx --model detection/inference/models/aoe2_yolo_v5.pt
"""

import argparse
from pathlib import Path


def export_to_onnx(model_path: str, imgsz: int = 640, dynamic: bool = True) -> str:
    """Export a YOLO .pt model to ONNX format.

    Args:
        model_path: Path to the .pt model file
        imgsz: Input image size (must match training size)
        dynamic: Dynamic batch + spatial axes. True enables batched SAHI and
            variable input sizes, but the ONNX Runtime **CoreML** provider cannot
            build a plan for a dynamic graph above ~640 px (it fails at 1280 with
            "Error in dynamically resizing"). Export with ``dynamic=False`` for a
            fixed-shape graph that CoreML/ANE serves at the given ``imgsz`` — the
            single-pass detector then runs on the Neural Engine at 1280. A static
            graph only accepts that one ``imgsz``, so every request must match it.

    Returns:
        Path to the exported .onnx file
    """
    from detection._ultralytics_compat import YOLO

    model = YOLO(model_path)
    mode = "dynamic" if dynamic else f"static {imgsz}px"
    print(f"Exporting {model_path} to ONNX (imgsz={imgsz}, {mode})...")

    export_path = model.export(  # pyright: ignore[reportAny]
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=dynamic,
    )

    print(f"Exported to: {export_path}")
    return str(export_path)


class _ExportOnnxArgs(argparse.Namespace):
    model: str
    imgsz: int
    dynamic: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent.parent / "inference" / "models" / "aoe2_yolo_v5.pt"),
        help="Path to .pt model file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dynamic axes (default). Use --no-dynamic for a fixed-shape graph "
        "the CoreML/ANE provider can serve at imgsz (required for 1280).",
    )
    args = parser.parse_args(namespace=_ExportOnnxArgs())

    export_to_onnx(args.model, args.imgsz, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
