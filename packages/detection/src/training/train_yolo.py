#!/usr/bin/env python3
"""
YOLO Training Script for AoE2 Detection

Trains YOLO models for AoE2 entity detection. Supports any ultralytics
base model (YOLO11, YOLO26, etc.).

Usage:
    python train_yolo.py --data <dataset>/dataset.yaml --name aoe2_yolo_v10
    python train_yolo.py --data <dataset>/dataset.yaml --name aoe2_yolo_v10 --resume

Defaults match the deployed contract: YOLO26n trained at 1280, and
``--export-onnx`` writes a static ``[1, 3, 1280, 1280]`` graph like v9's.
``--data`` and ``--name`` are required so a run never trains on a stale
dataset or overwrites an existing model's files.

Every run-defining setting is explicit. Ultralytics' ``optimizer=auto`` picks
the optimizer from the iteration count: v10, v11 and runs A/B/C/E/E-s got
AdamW(lr=0.000145, momentum=0.9), but run D crossed 10,000 iterations and got
MuSGD(lr=0.01), so adding data silently changed the learning rate ~70x. The
optimizer 2x2 with 2 seeds (R1, 2026-11-05) showed that AdamW at 0.000145
underfits: on the E data MuSGD scored F1 0.861 against 0.840 (p = 0.014), and
on the D data 0.869 against 0.860. The defaults are therefore MuSGD(lr=0.01,
momentum=0.9), what 'auto' chose for run D (v12); change them on purpose, not by
growing the dataset. Training never stops early and the exported model is the
final epoch (``last.pt``): the 8-frame validation set cannot pick checkpoints.

Cloud GPU training:
    1. Upload dataset: scp -r training_data_<version> root@<IP>:/workspace/
    2. Upload script:  scp train_yolo.py root@<IP>:/workspace/
    3. SSH and run:
    python train_yolo.py --data /workspace/training_data_<version>/dataset.yaml \
        --name aoe2_yolo_v10 --cache ram --export-onnx
    4. Download:
    scp root@<IP>:/workspace/runs/<name>/weights/last.pt ./

Estimated training time: ~1-2 hours on A100 for 150 epochs
"""

import argparse
from pathlib import Path


class _TrainYoloArgs(argparse.Namespace):
    data: str
    model: str
    epochs: int
    batch: int
    imgsz: int
    device: int
    workers: int
    cache: str
    patience: int
    optimizer: str
    lr0: float
    momentum: float
    warmup_bias_lr: float
    seed: int
    save_period: int
    project: str
    name: str
    resume: bool
    export_onnx: bool
    cls_gain: float | None
    box_gain: float | None
    dfl_gain: float | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO model for AoE2 detection")
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to dataset.yaml (absolute, or relative to packages/detection/src or cwd)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolo26n.pt",
        help="Base pretrained model (e.g., yolo26n.pt, yolo11n.pt). YOLO26 is NMS-free "
        "and ships STAL/ProgLoss for small objects — no extra train kwargs needed.",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=150, help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=16,
        help="Batch size (default: 16, adjust based on GPU memory)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Train and export size (default: 1280, the size the agent infers at)",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of data loader workers (default: 8)"
    )
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "none"],
        default="none",
        help="Cache images after epoch 1 for faster epochs: 'ram' (~14GB for the v8 set), "
        "'disk' (no RAM pressure), or 'none' (default). Pure I/O speedup — augmentation still "
        "runs fresh each epoch, so model quality is unaffected.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience; 0 (default) trains every epoch. The 8-frame "
        "validation set is too small to stop on.",
    )
    parser.add_argument(
        "--optimizer",
        default="MuSGD",
        help="Optimizer (default: MuSGD). Never 'auto': it picks AdamW at lr 0.000145, "
        "which underfits, until ceil(N/64)*epochs exceeds 10,000.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01, what 'auto' chose with MuSGD for v12)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum, or AdamW beta1 (default: 0.9, what 'auto' chose for v12)",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=0.0,
        help="Bias learning rate during warmup (default: 0.0, what 'auto' sets for both "
        "AdamW and MuSGD; Ultralytics' own default is 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Training seed (default: 0); vary it to measure noise"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save a checkpoint every N epochs (default: -1, off); 1 keeps the epochs "
        "that checkpoint averaging (SWA) needs",
    )
    parser.add_argument(
        "--project", default="runs", help="Project directory for outputs (default: runs)"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Run name and output model name, e.g. aoe2_yolo_v10 (must be a new version)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    # Loss-gain overrides (ultralytics defaults when omitted). Raising --cls-gain
    # biases toward classification accuracy on confusable cavalry lines (D2).
    parser.add_argument("--cls-gain", type=float, default=None, help="Classification loss gain")
    parser.add_argument("--box-gain", type=float, default=None, help="Box regression loss gain")
    parser.add_argument("--dfl-gain", type=float, default=None, help="Distribution focal loss gain")

    args = parser.parse_args(namespace=_TrainYoloArgs())

    # Import YOLO. In-repo, go through the compat shim (it centralizes the pyright
    # suppression for ultralytics' re-exported YOLO). On a bare machine — e.g. a
    # cloud GPU with only this file copied over — the `detection` package isn't
    # importable, so fall back to importing YOLO straight from ultralytics.
    try:
        from detection._ultralytics_compat import YOLO
    except ModuleNotFoundError:
        try:
            from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
        except ImportError as e:
            print(f"Error: ultralytics not installed ({e}). Install with: pip install ultralytics")
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
    print(f"AoE2 YOLO Training — {args.name}")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Optimizer: {args.optimizer} (lr0={args.lr0}, momentum={args.momentum})")
    print(f"Seed: {args.seed}")
    print(f"Device: cuda:{args.device}")
    print("=" * 60)

    # Ultralytics nests a relative project under its runs dir (e.g. 8.4.54 wrote
    # runs/detect/runs/<name>); an absolute project is used as given.
    project_dir = Path(args.project).resolve()

    # cache=ram/disk skips disk decode on epochs after the first; "none" -> False (no
    # caching). Augmentation still runs per-epoch, so this is a pure speedup.
    cache: str | bool = args.cache if args.cache != "none" else False

    last_model_path = project_dir / args.name / "weights" / "last.pt"
    if args.resume:
        # resume=True continues the interrupted run: same epoch count, optimizer
        # state and settings. Loading last.pt and calling train() without it
        # starts a new full run from those weights.
        if not last_model_path.exists():
            print(f"Error: no checkpoint to resume at {last_model_path}")
            return 1
        print(f"Resuming from: {last_model_path}")
        YOLO(str(last_model_path)).train(resume=True, workers=args.workers, cache=cache)
    else:
        model = YOLO(args.model)

        # Optional loss-gain overrides; omitted gains keep the ultralytics defaults.
        loss_gains: dict[str, float] = {
            name: value
            for name, value in (
                ("cls", args.cls_gain),
                ("box", args.box_gain),
                ("dfl", args.dfl_gain),
            )
            if value is not None
        }
        if loss_gains:
            print(f"Loss-gain overrides: {loss_gains}")

        # Train with optimized hyperparameters for AoE2
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=cache,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            momentum=args.momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            seed=args.seed,
            deterministic=True,
            save_period=args.save_period,
            **loss_gains,
            # Augmentation settings optimized for isometric game graphics
            hsv_h=0.015,  # Hue augmentation (slight)
            hsv_s=0.7,  # Saturation augmentation
            hsv_v=0.4,  # Value/brightness augmentation
            degrees=10,  # Rotation (units can face different directions)
            translate=0.1,  # Translation
            scale=0.5,  # Scale augmentation (important for zoom levels)
            flipud=0.0,  # No vertical flip (isometric view)
            fliplr=0.5,  # Horizontal flip OK
            mosaic=1.0,  # Mosaic augmentation
            mixup=0.1,  # MixUp augmentation
            # Output settings
            project=str(project_dir),
            name=args.name,
            exist_ok=True,
            verbose=True,
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print(f"Final model: {last_model_path}")
    if not last_model_path.exists():
        print(f"Error: final checkpoint not found at {last_model_path}")
        return 1

    # Export to ONNX if requested
    if args.export_onnx:
        # Static shape, like the deployed v9 graph: the ONNX Runtime CoreML
        # provider cannot plan a dynamic graph at 1280 (see export_onnx.py).
        print(f"\nExporting to ONNX (static {args.imgsz}px)...")
        final_model = YOLO(str(last_model_path))
        onnx_path_raw = final_model.export(  # pyright: ignore[reportAny]
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
        )
        onnx_path = str(onnx_path_raw)
        print(f"ONNX model: {onnx_path}")

        # Copy to inference/models using the run name. On a bare cloud machine
        # (only this file copied over) that directory does not exist; download
        # the run outputs instead.
        models_dir = script_dir / "inference" / "models"
        if not models_dir.is_dir():
            print(f"\n{models_dir} not found; download {last_model_path} and {onnx_path}")
            return 0

        import shutil

        dest_pt = models_dir / f"{args.name}.pt"
        dest_onnx = models_dir / f"{args.name}.onnx"

        shutil.copy(last_model_path, dest_pt)
        shutil.copy(onnx_path, dest_onnx)

        print("\nModels copied to:")
        print(f"  {dest_pt}")
        print(f"  {dest_onnx}")

    return 0


if __name__ == "__main__":
    exit(main())
