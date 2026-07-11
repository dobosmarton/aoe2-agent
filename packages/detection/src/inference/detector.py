"""
YOLO-based entity detection for AoE2.

Provides fast, accurate detection of game entities (units, buildings, resources)
with bounding boxes and semantic IDs for action targeting.

Supports both PyTorch (.pt) and ONNX (.onnx) model formats.
ONNX is recommended for Windows ARM64 where PyTorch is not available.
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from core import DetectedEntity

from ._ultralytics_results import yolo_boxes_to_lists
from .mock import mock_detect
from .onnx_layout import UnknownOnnxLayoutError, decode_example
from .postprocess import iou, nms
from .preprocess import letterbox
from .sahi import (
    compute_sahi_rois,
    merge_detections,
    onnx_sahi_detect,
    sahi_detect,
    sahi_detect_rois,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

    from .._classes_schema import ClassesYaml


def _load_default_classes() -> list[str]:
    """Load class names from classes.yaml (single source of truth).

    Falls back to a minimal list if the YAML file can't be loaded.
    PyTorch backend overrides this with model.names at load time anyway —
    this only affects ONNX and mock backends.
    """
    yaml_path = Path(__file__).parent.parent / "training" / "config" / "classes.yaml"
    try:
        import yaml

        with yaml_path.open() as f:
            data = cast("ClassesYaml", yaml.safe_load(f))
        classes = sorted(data["classes"], key=lambda c: c["id"])
        return [c["name"] for c in classes]
    except Exception:
        logger.warning("Could not load classes.yaml, using minimal fallback")
        return [
            "tree",
            "gold_mine",
            "stone_mine",
            "berry_bush",
            "relic",
            "deer",
            "boar",
            "wolf",
            "sheep",
            "town_center",
            "house",
            "lumber_camp",
            "mining_camp",
            "mill",
            "market",
            "dock",
            "farm",
            "barracks",
            "archery_range",
            "stable",
            "blacksmith",
            "siege_workshop",
            "monastery",
            "castle",
            "university",
            "gate",
            "wall",
            "tower",
            "wonder",
            "krepost",
            "villager",
            "trade_cart",
            "fishing_ship",
            "scout_line",
            "knight_line",
            "camel_line",
            "battle_elephant",
            "archer_line",
            "skirmisher_line",
            "cavalry_archer",
            "hand_cannoneer",
            "militia_line",
            "spearman_line",
            "eagle_line",
            "ram",
            "mangonel_line",
            "scorpion",
            "trebuchet",
            "monk",
            "king",
            "unique_archer",
            "unique_cavalry",
            "unique_infantry",
            "unique_siege",
            "unique_ship",
            "fish",
            "galley",
            "fire_galley",
            "siege_tower",
            "goose",
        ]


DEFAULT_CLASSES = _load_default_classes()


class EntityDetector:
    """YOLO-based entity detector for AoE2 screenshots.

    Supports PyTorch (.pt), ONNX (.onnx) models, and mock mode for testing.
    ONNX mode uses onnxruntime and works on Windows ARM64.
    """

    def __init__(
        self,
        model_path: str | None = None,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.35,
        use_mock: bool = False,
        imgsz: int = 1280,
        use_sahi: bool = True,
    ) -> None:
        """Initialize the detector.

        Args:
            model_path: Path to YOLO .pt or .onnx weights file
            class_names: List of class names (order matches model output)
            confidence_threshold: Minimum confidence for detections
            use_mock: If True, use mock detections (for testing without model)
            imgsz: Inference resolution (higher = more detections on large screenshots)
            use_sahi: If True, use SAHI sliced inference for large images (PyTorch only)
        """
        self.class_names = class_names or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        # Per-class thresholds from shared config
        from detection.inference.thresholds import CLASS_THRESHOLDS

        self.class_thresholds: dict[str, float] = dict(CLASS_THRESHOLDS)
        self.use_mock = use_mock
        self.use_sahi = use_sahi
        self.model = None
        self.onnx_session = None
        self.backend = None  # 'pytorch', 'onnx', or None
        self.input_size = imgsz  # YOLO input size
        # Populated from the ONNX graph in _load_onnx(). A static-shape export
        # (e.g. [1,3,640,640]) pins these; a dynamic re-export leaves them at
        # the permissive defaults so the configured imgsz / batched SAHI is used.
        self.onnx_batch_dynamic = True
        self.onnx_input_hw: int | None = None
        self._class_counters: dict[str, int] = {}
        # Persistent entity tracking across frames (fallback when tracker is None)
        self._previous_entities: list[DetectedEntity] = []
        self._global_id_counter: int = 0

        # Kalman filter tracker (replaces greedy IoU ID assignment)
        self.tracker = None
        try:
            from .tracker import EntityTracker

            self.tracker = EntityTracker()
        except Exception:
            logger.debug("Tracker not available, using greedy IoU ID assignment")

        if model_path and not use_mock:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load YOLO model from weights file (supports .pt and .onnx)."""
        path = Path(model_path)

        # Check for ONNX version if .pt specified but not found
        if not path.exists() and path.suffix == ".pt":
            onnx_path = path.with_suffix(".onnx")
            if onnx_path.exists():
                path = onnx_path
                model_path = str(onnx_path)

        if not path.exists():
            print(f"WARNING: Model not found: {model_path}. Using mock detection.")
            self.use_mock = True
            return

        # Load based on file extension
        if path.suffix == ".onnx":
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, model_path: str) -> None:
        """Load PyTorch YOLO model."""
        try:
            from detection._ultralytics_compat import YOLO

            self.model = YOLO(model_path)
            self.backend = "pytorch"
            self.use_mock = False
            # Use class names from the model itself (authoritative)
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = list(self.model.names.values())
            print(
                f"Loaded PyTorch model: {model_path} ({len(self.class_names)} classes, SAHI={'on' if self.use_sahi else 'off'})"
            )
        except ImportError:
            print("WARNING: ultralytics not installed. Trying ONNX...")
            # Try ONNX fallback
            onnx_path = Path(model_path).with_suffix(".onnx")
            if onnx_path.exists():
                self._load_onnx(str(onnx_path))
            else:
                print("WARNING: No ONNX model found. Using mock detection.")
                self.use_mock = True
        except Exception as e:
            print(f"WARNING: Failed to load PyTorch model: {e}. Using mock detection.")
            self.use_mock = True

    def _load_onnx(self, model_path: str) -> None:
        """Load ONNX model using onnxruntime with optimized session options."""
        try:
            import os

            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            num_threads = min(os.cpu_count() or 4, 8)
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = max(1, num_threads // 2)
            # Auto-detect best available provider
            available = ort.get_available_providers()
            providers = [
                p for p in ["DmlExecutionProvider", "CPUExecutionProvider"] if p in available
            ]
            self.onnx_session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.backend = "onnx"
            self.use_mock = False
            # Inspect the declared input shape so inference adapts to a
            # static-shape export instead of assuming a dynamic batch / the
            # configured imgsz. onnxruntime reports symbolic dims as strings.
            try:
                # onnxruntime types this loosely; absorb into object + narrow below.
                shape: list[object] = self.onnx_session.get_inputs()[0].shape
                if len(shape) == 4:
                    b, _c, h, w = shape
                    self.onnx_batch_dynamic = not (isinstance(b, int) and b > 0)
                    if isinstance(h, int) and h > 0 and isinstance(w, int) and w > 0:
                        self.onnx_input_hw = int(min(h, w))
                        if self.onnx_input_hw != self.input_size:
                            logger.info(
                                "ONNX has static input %sx%s; using it for inference (imgsz=%d ignored)",
                                h,
                                w,
                                self.input_size,
                            )
            except Exception:
                logger.debug("Could not read ONNX input shape; assuming dynamic", exc_info=True)
            print(
                f"Loaded ONNX model: {model_path} (providers={providers}, "
                f"static_hw={self.onnx_input_hw}, batch_dynamic={self.onnx_batch_dynamic})"
            )
        except ImportError:
            print("WARNING: onnxruntime not installed. Using mock detection.")
            self.use_mock = True
        except Exception as e:
            print(f"WARNING: Failed to load ONNX model: {e}. Using mock detection.")
            self.use_mock = True

    def _reset_counters(self) -> None:
        """Reset entity ID counters for new detection."""
        self._class_counters = dict.fromkeys(self.class_names, 0)

    def _generate_id(self, class_name: str) -> str:
        """Generate unique ID for detected entity."""
        if class_name not in self._class_counters:
            self._class_counters[class_name] = 0
        idx = self._class_counters[class_name]
        self._class_counters[class_name] += 1
        return f"{class_name}_{idx}"

    def _get_threshold(self, class_name: str) -> float:
        """Get confidence threshold for a class (lower for small entities)."""
        return self.class_thresholds.get(class_name, self.confidence_threshold)

    def _assign_persistent_ids(self, new_entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Match new detections to previous frame by IoU, preserving entity IDs.

        Entities that overlap >40% with a previous same-class entity keep the old ID.
        New entities get a globally unique ID that never repeats.
        """
        if not self._previous_entities:
            # First frame — assign fresh global IDs
            for entity in new_entities:
                entity.id = f"{entity.class_name}_{self._global_id_counter}"
                self._global_id_counter += 1
            self._previous_entities = new_entities
            return new_entities

        used_prev_indices: set[int] = set()

        for entity in new_entities:
            best_iou = 0.0
            best_idx = -1

            for i, prev in enumerate(self._previous_entities):
                if i in used_prev_indices or prev.class_name != entity.class_name:
                    continue
                cur_iou = iou(entity.bbox, prev.bbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = i

            if best_idx >= 0 and best_iou > 0.4:
                # Same entity — carry forward old ID
                entity.id = self._previous_entities[best_idx].id
                used_prev_indices.add(best_idx)
            else:
                # New entity — assign fresh global ID
                entity.id = f"{entity.class_name}_{self._global_id_counter}"
                self._global_id_counter += 1

        self._previous_entities = new_entities
        return new_entities

    def detect(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Detect entities in screenshot (full SAHI for accuracy).

        Args:
            screenshot: JPEG bytes or PIL Image

        Returns:
            List of detected entities sorted by class, then by confidence
        """
        import time

        t0 = time.monotonic()
        self._reset_counters()

        if self.use_mock:
            entities = mock_detect(screenshot, self._generate_id)
        elif self.backend == "onnx" and self.use_sahi:
            # ONNX + SAHI: batch all tiles in one inference call
            from PIL import Image as PILImage

            if isinstance(screenshot, bytes):
                image = PILImage.open(io.BytesIO(screenshot))
            else:
                image = screenshot
            if image.size[0] > 640:
                entities = onnx_sahi_detect(self, image)
            else:
                entities = self._onnx_detect(screenshot)
        elif self.backend == "onnx":
            entities = self._onnx_detect(screenshot)
        else:
            entities = self._pytorch_detect(screenshot)

        # Apply NMS to remove duplicate overlapping detections
        entities = nms(entities, iou_threshold=0.5)

        # Assign persistent IDs: Kalman tracker or greedy IoU fallback
        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info(
            "detect_full elapsed=%.2fs entities=%d mode=%s",
            elapsed,
            len(entities),
            "onnx_sahi" if self.backend == "onnx" and self.use_sahi else self.backend,
        )

        return entities

    def detect_fast(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Fast detection without SAHI tiling. For mid-turn rescans.

        Single-pass inference at imgsz resolution. Much faster (~1-2s vs ~28s)
        but may miss very small objects (~20px sheep). Use full detect() for
        the initial per-turn detection, and detect_fast() for rescans.
        """
        import time

        t0 = time.monotonic()
        self._reset_counters()

        if self.use_mock:
            entities = mock_detect(screenshot, self._generate_id)
        elif self.backend == "onnx":
            entities = self._onnx_detect(screenshot)
        else:
            from PIL import Image as PILImage

            if isinstance(screenshot, bytes):
                image = PILImage.open(io.BytesIO(screenshot))
            else:
                image = screenshot
            # Single-pass inference — skip SAHI
            if self.model is None:
                raise RuntimeError("Model not loaded; call _load_model() first")
            results = self.model(
                image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False
            )
            entities = self._parse_yolo_results(results)

        entities = nms(entities, iou_threshold=0.5)

        # Assign persistent IDs: Kalman tracker or greedy IoU fallback
        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info("detect_fast elapsed=%.2fs entities=%d", elapsed, len(entities))

        return entities

    def detect_fast_multi(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Multi-resolution fast detection for small object recovery.

        Two-pass inference without full SAHI tiling:
        1. Full image at input_size (1280) — catches buildings, trees, large entities
        2. Center 50% crop at 640 (native training res) — catches sheep, berries, deer

        ~2x cost of detect_fast() but catches small objects that single pass misses.
        Much faster than full SAHI (~2-3s vs ~28s).
        """
        import time

        from PIL import Image as PILImage

        t0 = time.monotonic()
        self._reset_counters()

        if isinstance(screenshot, bytes):
            image = PILImage.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        if self.use_mock:
            entities = mock_detect(screenshot, self._generate_id)
        elif self.backend == "onnx":
            # Pass 1: full image at input_size (1280)
            full_entities = self._onnx_detect(screenshot)

            # Pass 2: center 50% crop at 640 (native training resolution)
            w, h = image.size
            crop_x1 = w // 4
            crop_y1 = h // 4
            crop_x2 = w - crop_x1
            crop_y2 = h - crop_y1
            center_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Run at 640 — the model's native training resolution
            old_input_size = self.input_size
            self.input_size = 640
            crop_entities = self._onnx_detect(center_crop)
            self.input_size = old_input_size

            # Offset crop entities back to full image coordinates
            for e in crop_entities:
                x1, y1, x2, y2 = e.bbox
                e.bbox = (x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1)
                e.center = ((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2)

            entities = full_entities + crop_entities
            logger.debug(
                "detect_fast_multi: full=%d crop=%d", len(full_entities), len(crop_entities)
            )
        else:
            # PyTorch: same two-pass approach
            w, h = image.size
            if self.model is None:
                raise RuntimeError("Model not loaded; call _load_model() first")
            results = self.model(
                image,
                conf=min(self.class_thresholds.values()),
                imgsz=self.input_size,
                verbose=False,
            )
            full_entities = self._parse_yolo_results(results)

            crop_x1 = w // 4
            crop_y1 = h // 4
            crop_x2 = w - crop_x1
            crop_y2 = h - crop_y1
            center_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            if self.model is None:
                raise RuntimeError("Model not loaded; call _load_model() first")
            results = self.model(
                center_crop, conf=min(self.class_thresholds.values()), imgsz=640, verbose=False
            )
            crop_entities = self._parse_yolo_results(results)
            for e in crop_entities:
                x1, y1, x2, y2 = e.bbox
                e.bbox = (x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1)
                e.center = ((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2)

            entities = full_entities + crop_entities

        # NMS merges duplicates from both passes
        entities = nms(entities, iou_threshold=0.5)

        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info("detect_fast_multi elapsed=%.2fs entities=%d", elapsed, len(entities))

        return entities

    def detect_adaptive(
        self, screenshot: bytes | Image.Image, force_full: bool = False
    ) -> list[DetectedEntity]:
        """Adaptive detection: fast scan + targeted SAHI on entity clusters only.

        Reduces tile count from ~18 (full SAHI) to ~3-8 by running SAHI only
        on ROI regions around detected entities. Falls back to full SAHI on
        first turn, periodically, or when force_full=True.

        Args:
            screenshot: JPEG bytes or PIL Image
            force_full: If True, run full SAHI scan (e.g., first turn, alarm)
        """
        import time

        t0 = time.monotonic()
        self._reset_counters()

        # Fall back to full SAHI when we have no prior context
        if force_full or not self._previous_entities:
            return self.detect(screenshot)

        from PIL import Image as PILImage

        if isinstance(screenshot, bytes):
            image = PILImage.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        # 1. Fast single-pass scan (raw detections, no NMS/IDs yet)
        if self.use_mock:
            fast_entities = mock_detect(screenshot, self._generate_id)
        elif self.backend == "onnx":
            fast_entities = self._onnx_detect(screenshot)
        else:
            if self.model is None:
                raise RuntimeError("Model not loaded; call _load_model() first")
            results = self.model(
                image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False
            )
            fast_entities = self._parse_yolo_results(results)

        # 2. Compute ROI regions around entity clusters
        rois = compute_sahi_rois(fast_entities, self._previous_entities, image.size)

        if not rois:
            # No ROIs needed — fast scan is sufficient
            entities = nms(fast_entities, iou_threshold=0.5)
            if self.tracker:
                entities = self.tracker.update(entities)
            else:
                entities = self._assign_persistent_ids(entities)
            elapsed = time.monotonic() - t0
            logger.info(
                "detect_adaptive elapsed=%.2fs entities=%d rois=0 mode=fast_only",
                elapsed,
                len(entities),
            )
            return entities

        # 3. Check if adaptive is worth it — count ROI tiles vs full SAHI tiles
        tile_size = 640
        overlap = 32
        stride = tile_size - overlap
        roi_tile_count = 0
        for roi in rois:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            cols = max(1, -(-int(rx2 - rx1) // stride))
            rows = max(1, -(-int(ry2 - ry1) // stride))
            roi_tile_count += cols * rows
        w, h = image.size
        full_tile_count = max(1, -(-w // stride)) * max(1, -(-h // stride))
        if roi_tile_count >= full_tile_count * 0.7:
            logger.info(
                "adaptive_fallback roi_tiles=%d >= 70%% of full=%d", roi_tile_count, full_tile_count
            )
            return self.detect(screenshot)

        # Run SAHI only on ROI tiles
        sahi_entities = sahi_detect_rois(self, image, rois)

        # 4. Merge: keep fast entities outside ROIs, use SAHI inside ROIs
        merged = merge_detections(fast_entities, sahi_entities, rois)

        # 5. Final NMS + persistent IDs
        merged = nms(merged, iou_threshold=0.5)
        if self.tracker:
            merged = self.tracker.update(merged)
        else:
            merged = self._assign_persistent_ids(merged)

        elapsed = time.monotonic() - t0
        logger.info(
            "detect_adaptive elapsed=%.2fs entities=%d rois=%d mode=adaptive",
            elapsed,
            len(merged),
            len(rois),
        )
        return merged

    def _pytorch_detect(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Run detection using PyTorch/ultralytics backend.

        Uses SAHI sliced inference by default for large images to detect
        small entities (sheep, villagers) that get lost when downscaled.
        """
        from PIL import Image

        # Convert bytes to PIL Image if needed
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        # Use SAHI for large images (wider than tile size)
        if self.use_sahi and image.size[0] > 640:
            return sahi_detect(self, image)

        # Standard inference for small images
        if self.model is None:
            raise RuntimeError("Model not loaded; call _load_model() first")
        results = self.model(
            image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False
        )
        return self._parse_yolo_results(results)

    def _parse_yolo_results(self, results: object) -> list[DetectedEntity]:
        """Parse ultralytics YOLO results into DetectedEntity list."""
        entities = []
        for result in cast("list[object]", results):
            boxes_attr: object | None = getattr(result, "boxes", None)
            if boxes_attr is None:
                continue

            boxes, classes, confidences = yolo_boxes_to_lists(boxes_attr)
            for box, cls_id, conf in zip(boxes, classes, confidences, strict=True):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                class_idx = int(cls_id)

                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"unknown_{class_idx}"

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                entity = DetectedEntity(
                    id=self._generate_id(class_name),
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=(center_x, center_y),
                    confidence=float(conf),
                    area=area,
                )
                entities.append(entity)

        entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return entities

    def _onnx_detect(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Run detection using ONNX runtime backend."""
        from PIL import Image

        # Convert bytes to PIL Image if needed
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        # Store original size for scaling boxes back
        orig_width, orig_height = image.size

        # A static-shape ONNX export only accepts its declared input size; fall
        # back to the configured imgsz for a dynamic export.
        infer_size = self.onnx_input_hw or self.input_size

        # Letterbox (aspect-preserving) to match training preprocessing; a naive
        # square resize distorts sprites and loses small classes. See
        # inference/preprocess.py for the geometry and the inverse mapping below.
        box = letterbox(image, infer_size)
        img_array = np.expand_dims(box.chw, axis=0)

        # Run inference
        if self.onnx_session is None:
            raise RuntimeError("ONNX session not initialised; call _load_onnx() first")
        input_name = cast("str", self.onnx_session.get_inputs()[0].name)
        outputs = cast("list[np.ndarray]", self.onnx_session.run(None, {input_name: img_array}))

        raw_output = outputs[0]
        logger.debug("ONNX output shape: %s", raw_output.shape)

        min_confidence = (
            min(self.class_thresholds.values())
            if self.class_thresholds
            else self.confidence_threshold
        )
        try:
            rows = decode_example(cast("np.ndarray", raw_output[0]), min_confidence)
        except UnknownOnnxLayoutError:
            logger.warning(
                "Unrecognised ONNX output shape %s; returning no detections", raw_output.shape
            )
            return []

        # Undo the letterbox back to original screenshot coordinates:
        # orig = (coord - pad) / scale.
        inv = 1.0 / box.scale

        entities: list[DetectedEntity] = []
        for row in rows:
            class_name = (
                self.class_names[row.class_id]
                if row.class_id < len(self.class_names)
                else f"unknown_{row.class_id}"
            )
            if row.confidence < self._get_threshold(class_name):
                continue

            x1 = max(0.0, min((row.x1 - box.pad_x) * inv, orig_width))
            y1 = max(0.0, min((row.y1 - box.pad_y) * inv, orig_height))
            x2 = max(0.0, min((row.x2 - box.pad_x) * inv, orig_width))
            y2 = max(0.0, min((row.y2 - box.pad_y) * inv, orig_height))
            if x2 <= x1 or y2 <= y1:
                continue

            entities.append(
                DetectedEntity(
                    id=self._generate_id(class_name),
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                    confidence=row.confidence,
                    area=(x2 - x1) * (y2 - y1),
                )
            )

        entities.sort(key=lambda e: (e.class_name, -e.confidence))
        logger.debug("Final entity count: %d", len(entities))
        return entities

    def detect_to_dict_list(self, screenshot: bytes | Image.Image) -> list[dict]:
        """Detect and return as list of dictionaries.

        Convenience method for LLM context building.
        """
        return [e.to_dict() for e in self.detect(screenshot)]

    def find_entity_by_id(
        self, entities: list[DetectedEntity], target_id: str
    ) -> DetectedEntity | None:
        """Find an entity by its ID.

        Args:
            entities: List of detected entities
            target_id: Entity ID to find (e.g., "sheep_0")

        Returns:
            The matching entity, or None if not found
        """
        for entity in entities:
            if entity.id == target_id:
                return entity
        return None

    def find_entities_by_class(
        self, entities: list[DetectedEntity], class_name: str
    ) -> list[DetectedEntity]:
        """Find all entities of a given class.

        Args:
            entities: List of detected entities
            class_name: Class name to filter by

        Returns:
            List of matching entities (sorted by confidence)
        """
        matches = [e for e in entities if e.class_name == class_name]
        matches.sort(key=lambda e: -e.confidence)
        return matches

    def find_nearest_entity(
        self,
        entities: list[DetectedEntity],
        point: tuple[float, float],
        class_filter: str | None = None,
    ) -> DetectedEntity | None:
        """Find the nearest entity to a point.

        Args:
            entities: List of detected entities
            point: (x, y) coordinates
            class_filter: Optional class name to filter by

        Returns:
            Nearest entity, or None if no entities
        """
        candidates = entities
        if class_filter:
            candidates = self.find_entities_by_class(entities, class_filter)

        if not candidates:
            return None

        def distance(e: DetectedEntity) -> float:
            dx = float(e.center[0] - point[0])
            dy = float(e.center[1] - point[1])
            return math.sqrt(dx * dx + dy * dy)

        return min(candidates, key=distance)


# Singleton instance for easy access
_instance: EntityDetector | None = None


def resolve_model_path(model_name: str | None = None) -> str | None:
    """Resolve a model name to a weights file in the bundled models directory.

    The served version is configured in `apps/agent/src/config.py`
    (`detection_model` / AOE2_DETECTION_MODEL) — that is the single source of
    truth, passed down here as `model_name`. Callers without access to the
    agent config (tests, the remote-detector local fallback) pass no name and
    get the *highest-versioned* `aoe2_yolo_v*` weights present, so a stale
    hardcoded default can never silently resolve an old model again.

    ONNX is preferred over PyTorch (the ARM64 deploy path). Returns None when
    no matching weights exist (e.g. remote-only deploys where models are
    gitignored).
    """
    models_dir = Path(__file__).parent / "models"
    if model_name:
        for ext in ("onnx", "pt"):
            path = models_dir / f"{model_name}.{ext}"
            if path.exists():
                return str(path)
        return None

    def _version(path: Path) -> int:
        stem_version = path.stem.rsplit("_v", 1)[-1]
        return int(stem_version) if stem_version.isdigit() else -1

    candidates = [p for p in models_dir.glob("aoe2_yolo_v*") if p.suffix in (".onnx", ".pt")]
    if not candidates:
        return None
    # Highest version wins; at equal version ONNX beats PyTorch.
    best = max(candidates, key=lambda p: (_version(p), p.suffix == ".onnx"))
    return str(best)


def get_detector(
    model_path: str | None = None,
    use_mock: bool = False,
    imgsz: int = 1280,
    use_sahi: bool = True,
    model_name: str | None = None,
) -> EntityDetector:
    """Get or create the singleton detector instance.

    `model_name` names bundled weights (e.g. `"aoe2_yolo_v9"`) — the agent
    passes `config.detection_model` here so the served version has one source
    of truth. Without a name or explicit `model_path`, the newest bundled
    `aoe2_yolo_v*` weights are used (see `resolve_model_path`).

    `use_sahi=False` makes `detect()` run a single full-image pass instead of
    SAHI tiling — required for models whose training resolution doesn't match
    the SAHI tile scale (see EntityDetector docstring / testing/evaluate_real.py).
    """
    global _instance
    if _instance is None:
        if model_path is None:
            model_path = _resolve_or_substitute(model_name)
        _instance = EntityDetector(
            model_path=model_path, use_mock=use_mock, imgsz=imgsz, use_sahi=use_sahi
        )
    return _instance


def _resolve_or_substitute(model_name: str | None) -> str:
    """Weights path for `model_name`: resolved, substituted (loudly), or missing.

    Substitution: the configured model isn't bundled here (weights are
    gitignored; git pull never ships them) — use the newest bundled weights
    rather than degrading to mock, but say so LOUDLY: a silent substitution is
    how a v5 fallback served for a v9 config (2026-07-11 run review, F-5).
    """
    resolved = resolve_model_path(model_name)
    if resolved is not None:
        return resolved
    if model_name:
        substitute = resolve_model_path()
        if substitute is not None:
            logger.warning(
                "configured model %r not found in bundled weights; "
                "substituting %s — copy the configured weights here if this "
                "host should serve them",
                model_name,
                Path(substitute).name,
            )
            return substitute
    # Keep a deterministic (missing) path so EntityDetector's missing-model
    # handling degrades to mock with a warning. The name is a placeholder, not
    # a version — never hardcode a served version here (config.py owns that).
    missing = model_name or "aoe2_yolo_missing"
    return str(Path(__file__).parent / "models" / f"{missing}.onnx")


def current_detector() -> EntityDetector | None:
    """Return the existing singleton detector, or None — never creating one.

    Unlike `get_detector()`, this only peeks at an already-initialized detector, so
    callers that merely want to read live tracker state (e.g. entity velocities) can
    do so without accidentally loading a model on the remote/mock paths.
    """
    return _instance
