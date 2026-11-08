"""Export a YOLO checkpoint to ONNX for the detection server and agent.

The default is a static ``[1, 3, 1280, 1280]`` graph with the end-to-end head.
The served v12 export is rectangular and keeps the one-to-many head (host NMS);
copy the checkpoint to the served name first, because the ONNX is written
beside it with the same stem:

    cp aoe2_yolo_v12.pt aoe2_yolo_v12_nms736.pt
    python -m detection.training.export_onnx \
        --model packages/detection/src/inference/models/aoe2_yolo_v12_nms736.pt \
        --imgsz 736 1280 --no-end2end
"""

import argparse


def export_to_onnx(
    model_path: str,
    imgsz: int | tuple[int, int] = 1280,
    dynamic: bool = False,
    end2end: bool = True,
) -> str:
    """Export a YOLO .pt model to ONNX format.

    Args:
        model_path: Path to the .pt model file
        imgsz: Input side, or ``(height, width)``. The scale must match
            training: a 3024x1672 frame letterboxes to the same 0.42 scale at
            1280x1280 and at 736x1280, but 736x1280 computes 42% fewer pixels
            (CoreML on an M2 Pro: 10.0 ms against 15.3 ms).
        dynamic: Dynamic batch + spatial axes. True enables batched SAHI and
            variable input sizes, but the ONNX Runtime **CoreML** provider cannot
            build a plan for a dynamic graph above ~640 px (it fails at 1280 with
            "Error in dynamically resizing"). The default ``dynamic=False`` gives
            a fixed-shape graph that CoreML/ANE serves at the given ``imgsz``; the
            single-pass detector then runs on the Neural Engine at 1280. A static
            graph only accepts that one ``imgsz``; the server and the local
            detector read the shape from the graph.
        end2end: Keep the NMS-free one-to-one head (output ``[1, 300, 6]``).
            False exports the one-to-many head (output ``[1, 4+nc, anchors]``);
            `inference.onnx_layout` then runs a class-wise NMS. That head scored
            F1 0.876 against 0.867 for v12 on the independent test set, and
            CoreML runs all of its nodes in one partition.

    Returns:
        Path to the exported .onnx file
    """
    from detection._ultralytics_compat import YOLO

    model = YOLO(model_path)
    size = list(imgsz) if isinstance(imgsz, tuple) else imgsz
    mode = "dynamic" if dynamic else f"static {size}"
    head = "end-to-end" if end2end else "one-to-many, host NMS"
    print(f"Exporting {model_path} to ONNX (imgsz={size}, {mode}, {head})...")

    export_path = model.export(  # pyright: ignore[reportAny]
        format="onnx",
        imgsz=size,
        simplify=True,
        dynamic=dynamic,
        end2end=end2end,
    )

    print(f"Exported to: {export_path}")
    return str(export_path)


class _ExportOnnxArgs(argparse.Namespace):
    model: str
    imgsz: list[int]
    dynamic: bool
    end2end: bool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pt model file; the .onnx is written beside it",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[1280],
        metavar="SIZE",
        help="Input side, or HEIGHT WIDTH (e.g. 736 1280); the scale must match training",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dynamic axes. Off by default: the CoreML/ANE provider needs a "
        "fixed-shape graph at 1280.",
    )
    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the NMS-free end-to-end head. --no-end2end exports the "
        "one-to-many head; the server and agent then run NMS.",
    )
    args = parser.parse_args(argv, namespace=_ExportOnnxArgs())
    if len(args.imgsz) not in (1, 2):
        parser.error("--imgsz takes SIZE or HEIGHT WIDTH")

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else (args.imgsz[0], args.imgsz[1])
    export_to_onnx(args.model, imgsz, dynamic=args.dynamic, end2end=args.end2end)


if __name__ == "__main__":
    main()
