"""FastAPI server for YOLO inference on macOS via CoreML or ONNX.

Runs on the macOS host (Apple Silicon) to offload detection from the
Windows ARM64 VM. Accepts JPEG screenshots, returns JSON detections.

Usage:
    python -m server --model path/to/model.onnx
    python -m server --model path/to/model.mlpackage --port 8420
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Backend: TypeAlias = Literal["coreml", "onnx_coreml", "onnx_cpu"]

# ---------------------------------------------------------------------------
# Pydantic models (frozen — API boundary)
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_name: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) original image coords
    center: tuple[float, float]
    confidence: float
    area: float


class DetectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    detections: list[DetectionResult]
    inference_ms: float
    tile_count: int
    image_size: tuple[int, int]  # (width, height)


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    model_path: str
    backend: Backend
    num_classes: int


# ---------------------------------------------------------------------------
# Per-class confidence thresholds (single source of truth)
# ---------------------------------------------------------------------------

from detection.inference.thresholds import (
    CLASS_THRESHOLDS,
    DEFAULT_CONFIDENCE,
    get_threshold as _get_threshold,
)


# ---------------------------------------------------------------------------
# Class names loader
# ---------------------------------------------------------------------------

def _load_class_names() -> tuple[str, ...]:
    """Load class names from bundled classes.yaml."""
    yaml_path = Path(__file__).parent / "classes.yaml"
    try:
        import yaml

        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        classes = sorted(data["classes"], key=lambda c: c["id"])
        return tuple(c["name"] for c in classes)
    except (FileNotFoundError, KeyError, ValueError):
        logger.warning("Could not load classes.yaml, using hardcoded fallback")
        return (
            "tree", "gold_mine", "stone_mine", "berry_bush", "relic",
            "deer", "boar", "wolf", "sheep", "town_center", "house",
            "lumber_camp", "mining_camp", "mill", "market", "dock", "farm",
            "barracks", "archery_range", "stable", "blacksmith", "siege_workshop",
            "monastery", "castle", "university", "gate", "wall", "tower",
            "wonder", "krepost", "villager", "trade_cart", "fishing_ship",
            "scout_line", "knight_line", "camel_line", "battle_elephant",
            "archer_line", "skirmisher_line", "cavalry_archer", "hand_cannoneer",
            "militia_line", "spearman_line", "eagle_line", "ram", "mangonel_line",
            "scorpion", "trebuchet", "monk", "king", "unique_archer",
            "unique_cavalry", "unique_infantry", "unique_siege", "unique_ship",
            "fish", "galley", "fire_galley", "siege_tower", "goose",
        )


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelState:
    backend: Backend
    model: object  # ct.models.MLModel | ort.InferenceSession — narrowed via backend
    class_names: tuple[str, ...]
    model_path: str
    input_name: str = ""  # ONNX input tensor name


def _load_model(model_path: str) -> ModelState:
    """Load model with fallback chain: CoreML -> ONNX+CoreML EP -> ONNX CPU."""
    class_names = _load_class_names()
    path = Path(model_path)

    # Try CoreML native (.mlpackage or .mlmodel)
    if path.suffix in (".mlpackage", ".mlmodel") or path.is_dir():
        try:
            import coremltools as ct

            model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            logger.info("Loaded CoreML model: %s", model_path)
            return ModelState(
                backend="coreml",
                model=model,
                class_names=class_names,
                model_path=model_path,
            )
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("CoreML load failed: %s, trying ONNX fallback", e)

    # Try ONNX (with CoreML EP on macOS, or CPU)
    if path.suffix == ".onnx" or not path.suffix:
        onnx_path = path if path.suffix == ".onnx" else path.with_suffix(".onnx")
        if onnx_path.exists():
            return _load_onnx(str(onnx_path), class_names)

    # Last resort: look for .onnx sibling
    onnx_sibling = path.with_suffix(".onnx")
    if onnx_sibling.exists():
        return _load_onnx(str(onnx_sibling), class_names)

    raise FileNotFoundError(f"No loadable model found at {model_path}")


def _load_onnx(onnx_path: str, class_names: tuple[str, ...]) -> ModelState:
    """Load ONNX model with best available provider."""
    import os

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    num_threads = min(os.cpu_count() or 4, 8)
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = max(1, num_threads // 2)

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        backend: Backend = "onnx_coreml"
    else:
        providers = ["CPUExecutionProvider"]
        backend = "onnx_cpu"

    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    logger.info("Loaded ONNX model: %s (providers=%s)", onnx_path, providers)

    return ModelState(
        backend=backend,
        model=session,
        class_names=class_names,
        model_path=onnx_path,
        input_name=input_name,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _preprocess(image: Image.Image, target_size: int) -> tuple[np.ndarray, int, int]:
    """Resize + normalize a PIL Image for YOLO inference.

    Returns (chw_array, orig_width, orig_height).
    chw_array shape: (3, target_size, target_size) float32 [0, 1].
    """
    orig_w, orig_h = image.size
    resized = image.resize((target_size, target_size))
    arr = np.array(resized, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return chw, orig_w, orig_h


def _run_onnx_batch(state: ModelState, batch: np.ndarray) -> np.ndarray:
    """Run batched ONNX inference. batch shape: (N, 3, H, W)."""
    outputs = state.model.run(None, {state.input_name: batch})  # type: ignore[union-attr]
    return outputs[0]


def _run_coreml_single(state: ModelState, chw: np.ndarray) -> np.ndarray:
    """Run single-image CoreML inference. chw shape: (3, H, W)."""
    spec = state.model.get_spec()  # type: ignore[union-attr]
    input_desc = spec.description.input[0]
    input_key = input_desc.name

    # CoreML models can expect either PIL Image or numpy array
    input_type = input_desc.type.WhichOneof("Type")
    if input_type == "imageType":
        from PIL import Image as PILImage
        hwc = np.transpose(chw, (1, 2, 0))  # CHW → HWC
        hwc_uint8 = (hwc * 255).astype(np.uint8)
        input_data = PILImage.fromarray(hwc_uint8, mode="RGB")
    else:
        input_data = np.expand_dims(chw, axis=0)  # (1, 3, H, W)

    prediction = state.model.predict({input_key: input_data})  # type: ignore[union-attr]

    output_key = spec.description.output[0].name
    result = prediction[output_key]

    if not isinstance(result, np.ndarray):
        result = np.array(result)
    if result.ndim == 2:
        result = np.expand_dims(result, axis=0)

    return result


def _parse_raw_output(
    raw: np.ndarray,
    batch_idx: int,
    num_classes: int,
    class_names: tuple[str, ...],
    scale_x: float,
    scale_y: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    clip_w: float | None = None,
    clip_h: float | None = None,
) -> list[DetectionResult]:
    """Parse ONNX/CoreML raw output for one image in the batch.

    Handles both output formats:
    - Post-NMS: (batch, num_detections, 6)
    - Raw: (batch, 4+num_classes, num_boxes)
    """
    min_thresh = min(CLASS_THRESHOLDS.values())

    if len(raw.shape) == 3 and raw.shape[2] == 6:
        predictions = raw[batch_idx]
    elif len(raw.shape) == 3 and raw.shape[1] == (4 + num_classes):
        preds_raw = raw[batch_idx].T  # (num_boxes, 4+num_classes)
        boxes = preds_raw[:, :4]
        class_scores = preds_raw[:, 4:]
        best_cls = np.argmax(class_scores, axis=1)
        best_conf = np.max(class_scores, axis=1)
        mask = best_conf >= min_thresh
        boxes, best_cls, best_conf = boxes[mask], best_cls[mask], best_conf[mask]
        preds = []
        for box, cls_id, conf in zip(boxes, best_cls, best_conf):
            xc, yc, w, h = box
            preds.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, conf, cls_id])
        predictions = np.array(preds) if preds else np.array([]).reshape(0, 6)
    else:
        return []

    results: list[DetectionResult] = []
    for pred in predictions:
        x1, y1, x2, y2, confidence, class_id = pred
        class_idx = int(class_id)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"unknown_{class_idx}"

        if confidence < _get_threshold(class_name):
            continue

        abs_x1 = x1 * scale_x + x_offset
        abs_y1 = y1 * scale_y + y_offset
        abs_x2 = x2 * scale_x + x_offset
        abs_y2 = y2 * scale_y + y_offset

        if clip_w is not None:
            abs_x2 = min(abs_x2, x_offset + clip_w)
        if clip_h is not None:
            abs_y2 = min(abs_y2, y_offset + clip_h)

        if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
            continue

        results.append(DetectionResult(
            class_name=class_name,
            bbox=(float(abs_x1), float(abs_y1), float(abs_x2), float(abs_y2)),
            center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
            confidence=float(confidence),
            area=float((abs_x2 - abs_x1) * (abs_y2 - abs_y1)),
        ))

    return results


def _detect_single(state: ModelState, image: Image.Image, imgsz: int) -> list[DetectionResult]:
    """Single-image detection (no tiling)."""
    chw, orig_w, orig_h = _preprocess(image, imgsz)
    num_classes = len(state.class_names)

    if state.backend == "coreml":
        raw = _run_coreml_single(state, chw)
    else:
        batch = np.expand_dims(chw, axis=0)
        raw = _run_onnx_batch(state, batch)

    scale_x = orig_w / imgsz
    scale_y = orig_h / imgsz
    return _parse_raw_output(raw, 0, num_classes, state.class_names, scale_x, scale_y)


def _detect_sahi(
    state: ModelState,
    image: Image.Image,
    tile_size: int = 640,
    overlap: int = 32,
) -> tuple[list[DetectionResult], int]:
    """SAHI tiled detection. Returns (detections, tile_count)."""
    from PIL import Image as PILImage

    width, height = image.size
    stride = tile_size - overlap
    num_classes = len(state.class_names)

    # Collect tiles
    tiles: list[np.ndarray] = []
    offsets: list[tuple[int, int, int, int]] = []  # (x_start, y_start, tile_w, tile_h)

    for y_start in range(0, height, stride):
        for x_start in range(0, width, stride):
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            tile = image.crop((x_start, y_start, x_end, y_end))
            tile_w, tile_h = tile.size

            if tile.size != (tile_size, tile_size):
                padded = PILImage.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            arr = np.array(tile, dtype=np.float32) / 255.0
            chw = np.transpose(arr, (2, 0, 1))
            tiles.append(chw)
            offsets.append((x_start, y_start, tile_w, tile_h))

    if not tiles:
        return [], 0

    all_detections: list[DetectionResult] = []

    if state.backend == "coreml":
        # CoreML: sequential per-tile (no dynamic batch support)
        for chw, (x_off, y_off, tw, th) in zip(tiles, offsets):
            raw = _run_coreml_single(state, chw)
            dets = _parse_raw_output(
                raw, 0, num_classes, state.class_names,
                scale_x=1.0, scale_y=1.0,
                x_offset=x_off, y_offset=y_off,
                clip_w=tw, clip_h=th,
            )
            all_detections.extend(dets)
    else:
        # ONNX: batch all tiles in one call
        batch = np.stack(tiles, axis=0)
        raw = _run_onnx_batch(state, batch)

        for tile_idx, (x_off, y_off, tw, th) in enumerate(offsets):
            dets = _parse_raw_output(
                raw, tile_idx, num_classes, state.class_names,
                scale_x=1.0, scale_y=1.0,
                x_offset=x_off, y_offset=y_off,
                clip_w=tw, clip_h=th,
            )
            all_detections.extend(dets)

    return all_detections, len(tiles)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_model_state: ModelState | None = None


def create_app(model_path: str) -> FastAPI:
    """Create FastAPI app with model loaded at startup."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        global _model_state
        _model_state = _load_model(model_path)
        logger.info(
            "Server ready: backend=%s, classes=%d, model=%s",
            _model_state.backend,
            len(_model_state.class_names),
            _model_state.model_path,
        )
        yield
        _model_state = None

    app = FastAPI(title="AoE2 YOLO Detection Server", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        assert _model_state is not None
        return HealthResponse(
            status="ok",
            model_path=_model_state.model_path,
            backend=_model_state.backend,
            num_classes=len(_model_state.class_names),
        )

    @app.post("/detect", response_model=DetectionResponse)
    async def detect(
        file: UploadFile = File(...),
        imgsz: int = Query(default=640, ge=320, le=1920),
    ) -> DetectionResponse:
        """Single-image detection (no SAHI tiling)."""
        import asyncio

        from PIL import Image as PILImage

        assert _model_state is not None
        data = await file.read()
        image = PILImage.open(io.BytesIO(data)).convert("RGB")
        width, height = image.size

        t0 = time.monotonic()
        detections = await asyncio.to_thread(_detect_single, _model_state, image, imgsz)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return DetectionResponse(
            detections=detections,
            inference_ms=round(elapsed_ms, 1),
            tile_count=1,
            image_size=(width, height),
        )

    @app.post("/detect/sahi", response_model=DetectionResponse)
    async def detect_sahi(
        file: UploadFile = File(...),
        tile_size: int = Query(default=640, ge=320, le=1280),
        overlap: int = Query(default=32, ge=0, le=128),
    ) -> DetectionResponse:
        """SAHI tiled detection for full-accuracy scans."""
        import asyncio

        from PIL import Image as PILImage

        assert _model_state is not None
        data = await file.read()
        image = PILImage.open(io.BytesIO(data)).convert("RGB")
        width, height = image.size

        t0 = time.monotonic()
        detections, tile_count = await asyncio.to_thread(
            _detect_sahi, _model_state, image, tile_size, overlap,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        return DetectionResponse(
            detections=detections,
            inference_ms=round(elapsed_ms, 1),
            tile_count=tile_count,
            image_size=(width, height),
        )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="AoE2 YOLO Detection Server (CoreML/ONNX)")
    parser.add_argument("--model", required=True, help="Path to .mlpackage or .onnx model")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8420, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
