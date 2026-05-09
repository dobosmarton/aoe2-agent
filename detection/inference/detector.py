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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .mock import mock_detect
from .postprocess import iou, nms
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


@dataclass
class DetectedEntity:
    """Represents a detected game entity."""

    id: str  # Unique ID: e.g., "sheep_0", "villager_1"
    class_name: str  # Entity class: "sheep", "villager", "tc"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]  # Center point (x, y)
    confidence: float  # Detection confidence 0-1
    area: float = field(default=0)  # Bounding box area in pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox),
            "center": self.center,
            "confidence": self.confidence,
        }


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
            data = yaml.safe_load(f)
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
    ):
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

    def _load_model(self, model_path: str):
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

    def _load_pytorch(self, model_path: str):
        """Load PyTorch YOLO model."""
        try:
            from ultralytics import YOLO

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

    def _load_onnx(self, model_path: str):
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
            print(f"Loaded ONNX model: {model_path} (providers={providers})")
        except ImportError:
            print("WARNING: onnxruntime not installed. Using mock detection.")
            self.use_mock = True
        except Exception as e:
            print(f"WARNING: Failed to load ONNX model: {e}. Using mock detection.")
            self.use_mock = True

    def _reset_counters(self):
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
        results = self.model(
            image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False
        )
        return self._parse_yolo_results(results)

    def _parse_yolo_results(self, results) -> list[DetectedEntity]:
        """Parse ultralytics YOLO results into DetectedEntity list."""
        entities = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                strict=True,
            ):
                x1, y1, x2, y2 = box.tolist()
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

        # Preprocess: resize and normalize for YOLO
        image_resized = image.resize((self.input_size, self.input_size))
        img_array = np.array(image_resized).astype(np.float32) / 255.0

        # Convert from HWC to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img_array})

        # Debug: print output shape to understand format
        raw_output = outputs[0]
        logger.debug("ONNX output shape: %s", raw_output.shape)

        # Handle different ONNX output formats from ultralytics
        # Format 1: Post-NMS (1, num_detections, 6) = x1, y1, x2, y2, conf, class_id
        # Format 2: Raw predictions (1, 4+num_classes, num_boxes) = needs transposing and NMS

        if len(raw_output.shape) == 3 and raw_output.shape[2] == 6:
            # Post-NMS format: (1, num_detections, 6)
            predictions = raw_output[0]
            logger.debug("Post-NMS format, %d detection slots", len(predictions))

            if logger.isEnabledFor(logging.DEBUG):
                confidences = predictions[:, 4]
                non_zero = confidences[confidences > 0.01]
                logger.debug("Confidences > 0.01: %d", len(non_zero))
                if len(non_zero) > 0:
                    logger.debug("Max conf: %.4f, Min conf: %.4f", non_zero.max(), non_zero.min())
        elif len(raw_output.shape) == 3 and raw_output.shape[1] == (4 + len(self.class_names)):
            # Raw format: (1, 4+num_classes, num_boxes) - needs transposing
            # Shape is (1, 50, 8400) for 46 classes -> transpose to (8400, 50)
            predictions_raw = raw_output[0].T  # Now (num_boxes, 4+num_classes)
            logger.debug("Raw format, %d boxes, processing...", len(predictions_raw))

            # Extract boxes, scores, and class predictions
            boxes = predictions_raw[:, :4]  # x_center, y_center, width, height
            class_scores = predictions_raw[:, 4:]  # (num_boxes, num_classes)

            # Get best class and confidence for each box
            best_class_idx = np.argmax(class_scores, axis=1)
            best_confidence = np.max(class_scores, axis=1)

            # Filter by confidence (use lowest per-class threshold for bulk filter)
            min_thresh = (
                min(self.class_thresholds.values())
                if self.class_thresholds
                else self.confidence_threshold
            )
            mask = best_confidence >= min_thresh
            boxes = boxes[mask]
            best_class_idx = best_class_idx[mask]
            best_confidence = best_confidence[mask]

            logger.debug("%d boxes after confidence filter (%.2f)", len(boxes), min_thresh)

            # Convert from x_center, y_center, w, h to x1, y1, x2, y2
            predictions = []
            for box, cls_id, conf in zip(boxes, best_class_idx, best_confidence, strict=True):
                x_c, y_c, w, h = box
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2
                predictions.append([x1, y1, x2, y2, conf, cls_id])
            predictions = np.array(predictions) if predictions else np.array([]).reshape(0, 6)
        else:
            logger.debug("Unknown format shape %s, trying as post-NMS", raw_output.shape)
            predictions = raw_output[0] if len(raw_output.shape) == 3 else raw_output

        # Scale factors for converting from 640x640 to original size
        scale_x = orig_width / self.input_size
        scale_y = orig_height / self.input_size

        entities = []
        for pred in predictions:
            x1, y1, x2, y2, confidence, class_id = pred

            # Per-class confidence threshold
            class_idx = int(class_id)
            class_name = (
                self.class_names[class_idx]
                if class_idx < len(self.class_names)
                else f"unknown_{class_idx}"
            )
            if confidence < self._get_threshold(class_name):
                continue

            # Scale coordinates back to original image size
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y

            # Clamp to image bounds
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Calculate center and area
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            entity = DetectedEntity(
                id=self._generate_id(class_name),
                class_name=class_name,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float(center_x), float(center_y)),
                confidence=float(confidence),
                area=float(area),
            )
            entities.append(entity)

        # Sort by class name, then by confidence (highest first)
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
            dx = e.center[0] - point[0]
            dy = e.center[1] - point[1]
            return (dx * dx + dy * dy) ** 0.5

        return min(candidates, key=distance)


# Singleton instance for easy access
_instance: EntityDetector | None = None


def get_detector(
    model_path: str | None = None,
    use_mock: bool = False,
    imgsz: int = 1280,
) -> EntityDetector:
    """Get or create the singleton detector instance.

    Model priority (from highest to lowest):
    1. Explicitly provided model_path
    2. v6 ONNX/PT (aoe2_yolo_v6) - YOLO26, NMS-free
    3. v5 ONNX/PT (aoe2_yolo_v5) - YOLO11, batched SAHI
    4. v2 model (aoe2_yolo_v2) - hybrid trained
    5. v1 model (aoe2_yolo26) - synthetic only, fallback
    """
    global _instance
    if _instance is None:
        if model_path is None:
            models_dir = Path(__file__).parent / "models"

            # v6 model (YOLO26 - NMS-free, fastest)
            v6_onnx_path = models_dir / "aoe2_yolo_v6.onnx"
            v6_pt_path = models_dir / "aoe2_yolo_v6.pt"

            # v5 model (YOLO11 - current production)
            v5_onnx_path = models_dir / "aoe2_yolo_v5.onnx"
            v5_pt_path = models_dir / "aoe2_yolo_v5.pt"

            # v2 model (hybrid training - fallback)
            v2_onnx_path = models_dir / "aoe2_yolo_v2.onnx"
            v2_pt_path = models_dir / "aoe2_yolo_v2.pt"

            # v1 model (synthetic only - last resort)
            v1_onnx_path = models_dir / "aoe2_yolo26.onnx"
            v1_pt_path = models_dir / "aoe2_yolo26.pt"

            # Priority: v6 > v5 > v2 > v1 (ONNX preferred over PT)
            if v6_onnx_path.exists():
                model_path = str(v6_onnx_path)
            elif v6_pt_path.exists():
                model_path = str(v6_pt_path)
            elif v5_onnx_path.exists():
                model_path = str(v5_onnx_path)
            elif v5_pt_path.exists():
                model_path = str(v5_pt_path)
            elif v2_onnx_path.exists():
                model_path = str(v2_onnx_path)
            elif v2_pt_path.exists():
                model_path = str(v2_pt_path)
            elif v1_onnx_path.exists():
                model_path = str(v1_onnx_path)
            elif v1_pt_path.exists():
                model_path = str(v1_pt_path)
            else:
                model_path = str(v5_pt_path)  # Will fail gracefully

        _instance = EntityDetector(model_path=model_path, use_mock=use_mock, imgsz=imgsz)
    return _instance
