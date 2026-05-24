"""Export YOLO model to CoreML format for Apple Neural Engine inference.

Usage:
    just export-coreml
    python -m detection.training.export_coreml --model detection/inference/models/aoe2_yolo_v5.pt

Then serve with:
    just server --model detection/inference/models/aoe2_yolo_v5.mlpackage
"""

import argparse
from pathlib import Path


def export_to_coreml(model_path: str, imgsz: int = 640) -> str:
    """Export a YOLO .pt model to CoreML (.mlpackage) format.

    Args:
        model_path: Path to the .pt model file
        imgsz: Input image size (must match training size)

    Returns:
        Path to the exported .mlpackage directory
    """
    from detection._ultralytics_compat import YOLO

    model = YOLO(model_path)
    print(f"Exporting {model_path} to CoreML (imgsz={imgsz}, nms=False)...")

    export_path = model.export(  # pyright: ignore[reportAny]
        format="coreml",
        imgsz=imgsz,
        nms=False,  # NMS runs client-side for flexibility
    )

    print(f"Exported to: {export_path}")
    return str(export_path)


class _ExportCoreMLArgs(argparse.Namespace):
    model: str
    imgsz: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model to CoreML")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent.parent / "inference" / "models" / "aoe2_yolo_v5.pt"),
        help="Path to .pt model file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args(namespace=_ExportCoreMLArgs())

    export_to_coreml(args.model, args.imgsz)


if __name__ == "__main__":
    main()
