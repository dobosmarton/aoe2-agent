"""
YOLO-based entity detection for AoE2.

Provides fast, accurate detection of game entities (units, buildings, resources)
with bounding boxes and semantic IDs for action targeting.

Supports both PyTorch (.pt) and ONNX (.onnx) model formats.
ONNX is recommended for Windows ARM64 where PyTorch is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import io
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class DetectedEntity:
    """Represents a detected game entity."""
    id: str                          # Unique ID: e.g., "sheep_0", "villager_1"
    class_name: str                  # Entity class: "sheep", "villager", "tc"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]      # Center point (x, y)
    confidence: float                # Detection confidence 0-1
    area: float = field(default=0)   # Bounding box area in pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox),
            "center": self.center,
            "confidence": self.confidence
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
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        classes = sorted(data["classes"], key=lambda c: c["id"])
        return [c["name"] for c in classes]
    except Exception:
        logger.warning("Could not load classes.yaml, using minimal fallback")
        return [
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
        ]


DEFAULT_CLASSES = _load_default_classes()


class EntityDetector:
    """YOLO-based entity detector for AoE2 screenshots.

    Supports PyTorch (.pt), ONNX (.onnx) models, and mock mode for testing.
    ONNX mode uses onnxruntime and works on Windows ARM64.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names: Optional[list[str]] = None,
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
        # Lower thresholds for small/hard-to-detect entities
        self.class_thresholds: dict[str, float] = {
            "sheep": 0.20, "deer": 0.20, "berry_bush": 0.25,
            "villager": 0.25, "relic": 0.20,
        }
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
        if not path.exists() and path.suffix == '.pt':
            onnx_path = path.with_suffix('.onnx')
            if onnx_path.exists():
                path = onnx_path
                model_path = str(onnx_path)

        if not path.exists():
            print(f"WARNING: Model not found: {model_path}. Using mock detection.")
            self.use_mock = True
            return

        # Load based on file extension
        if path.suffix == '.onnx':
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, model_path: str):
        """Load PyTorch YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.backend = 'pytorch'
            self.use_mock = False
            # Use class names from the model itself (authoritative)
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            print(f"Loaded PyTorch model: {model_path} ({len(self.class_names)} classes, SAHI={'on' if self.use_sahi else 'off'})")
        except ImportError:
            print("WARNING: ultralytics not installed. Trying ONNX...")
            # Try ONNX fallback
            onnx_path = Path(model_path).with_suffix('.onnx')
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
            providers = [p for p in ['DmlExecutionProvider', 'CPUExecutionProvider']
                         if p in available]
            self.onnx_session = ort.InferenceSession(
                model_path, sess_options, providers=providers
            )
            self.backend = 'onnx'
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
        self._class_counters = {name: 0 for name in self.class_names}

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
                iou = self._iou(entity.bbox, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
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

    def detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
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
            entities = self._mock_detect(screenshot)
        elif self.backend == 'onnx' and self.use_sahi:
            # ONNX + SAHI: batch all tiles in one inference call
            from PIL import Image as PILImage
            if isinstance(screenshot, bytes):
                image = PILImage.open(io.BytesIO(screenshot))
            else:
                image = screenshot
            if image.size[0] > 640:
                entities = self._onnx_sahi_detect(image)
            else:
                entities = self._onnx_detect(screenshot)
        elif self.backend == 'onnx':
            entities = self._onnx_detect(screenshot)
        else:
            entities = self._pytorch_detect(screenshot)

        # Apply NMS to remove duplicate overlapping detections
        entities = self._nms(entities, iou_threshold=0.5)

        # Assign persistent IDs: Kalman tracker or greedy IoU fallback
        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info("detect_full elapsed=%.2fs entities=%d mode=%s",
                     elapsed, len(entities),
                     "onnx_sahi" if self.backend == 'onnx' and self.use_sahi else self.backend)

        return entities

    def detect_fast(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
        """Fast detection without SAHI tiling. For mid-turn rescans.

        Single-pass inference at imgsz resolution. Much faster (~1-2s vs ~28s)
        but may miss very small objects (~20px sheep). Use full detect() for
        the initial per-turn detection, and detect_fast() for rescans.
        """
        import time
        t0 = time.monotonic()
        self._reset_counters()

        if self.use_mock:
            entities = self._mock_detect(screenshot)
        elif self.backend == 'onnx':
            entities = self._onnx_detect(screenshot)
        else:
            from PIL import Image as PILImage
            if isinstance(screenshot, bytes):
                image = PILImage.open(io.BytesIO(screenshot))
            else:
                image = screenshot
            # Single-pass inference — skip SAHI
            results = self.model(image, conf=self.confidence_threshold,
                                 imgsz=self.input_size, verbose=False)
            entities = self._parse_yolo_results(results)

        entities = self._nms(entities, iou_threshold=0.5)

        # Assign persistent IDs: Kalman tracker or greedy IoU fallback
        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info("detect_fast elapsed=%.2fs entities=%d", elapsed, len(entities))

        return entities

    def detect_fast_multi(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
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
            entities = self._mock_detect(screenshot)
        elif self.backend == 'onnx':
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
            logger.debug("detect_fast_multi: full=%d crop=%d", len(full_entities), len(crop_entities))
        else:
            # PyTorch: same two-pass approach
            w, h = image.size
            results = self.model(image, conf=min(self.class_thresholds.values()),
                                 imgsz=self.input_size, verbose=False)
            full_entities = self._parse_yolo_results(results)

            crop_x1 = w // 4
            crop_y1 = h // 4
            crop_x2 = w - crop_x1
            crop_y2 = h - crop_y1
            center_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            results = self.model(center_crop, conf=min(self.class_thresholds.values()),
                                 imgsz=640, verbose=False)
            crop_entities = self._parse_yolo_results(results)
            for e in crop_entities:
                x1, y1, x2, y2 = e.bbox
                e.bbox = (x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1)
                e.center = ((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2)

            entities = full_entities + crop_entities

        # NMS merges duplicates from both passes
        entities = self._nms(entities, iou_threshold=0.5)

        if self.tracker:
            entities = self.tracker.update(entities)
        else:
            entities = self._assign_persistent_ids(entities)

        elapsed = time.monotonic() - t0
        logger.info("detect_fast_multi elapsed=%.2fs entities=%d", elapsed, len(entities))

        return entities

    def detect_adaptive(
        self, screenshot: Union[bytes, "Image.Image"], force_full: bool = False
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
            fast_entities = self._mock_detect(screenshot)
        elif self.backend == 'onnx':
            fast_entities = self._onnx_detect(screenshot)
        else:
            results = self.model(image, conf=self.confidence_threshold,
                                 imgsz=self.input_size, verbose=False)
            fast_entities = self._parse_yolo_results(results)

        # 2. Compute ROI regions around entity clusters
        rois = self._compute_sahi_rois(fast_entities, self._previous_entities, image.size)

        if not rois:
            # No ROIs needed — fast scan is sufficient
            entities = self._nms(fast_entities, iou_threshold=0.5)
            if self.tracker:
                entities = self.tracker.update(entities)
            else:
                entities = self._assign_persistent_ids(entities)
            elapsed = time.monotonic() - t0
            logger.info("detect_adaptive elapsed=%.2fs entities=%d rois=0 mode=fast_only",
                         elapsed, len(entities))
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
            logger.info("adaptive_fallback roi_tiles=%d >= 70%% of full=%d",
                        roi_tile_count, full_tile_count)
            return self.detect(screenshot)

        # Run SAHI only on ROI tiles
        sahi_entities = self._sahi_detect_rois(image, rois)

        # 4. Merge: keep fast entities outside ROIs, use SAHI inside ROIs
        merged = self._merge_detections(fast_entities, sahi_entities, rois)

        # 5. Final NMS + persistent IDs
        merged = self._nms(merged, iou_threshold=0.5)
        if self.tracker:
            merged = self.tracker.update(merged)
        else:
            merged = self._assign_persistent_ids(merged)

        elapsed = time.monotonic() - t0
        logger.info("detect_adaptive elapsed=%.2fs entities=%d rois=%d mode=adaptive",
                     elapsed, len(merged), len(rois))
        return merged

    def _compute_sahi_rois(
        self,
        fast_entities: list[DetectedEntity],
        previous_entities: list[DetectedEntity],
        image_size: tuple[int, int],
    ) -> list[tuple[float, float, float, float]]:
        """Compute ROI regions for targeted SAHI based on entity clusters.

        Groups entities within 200px into clusters, adds 128px padding,
        and merges overlapping ROIs. Also includes regions where previous
        entities disappeared (may have moved just beyond fast-pass range).
        """
        # Collect bboxes from current detections + disappeared previous entities
        all_bboxes = [e.bbox for e in fast_entities]

        for prev in previous_entities:
            # Include previous entities not found in fast scan
            if not any(self._iou(prev.bbox, f.bbox) > 0.3 for f in fast_entities):
                all_bboxes.append(prev.bbox)

        if not all_bboxes:
            return []

        # Union-Find clustering: group entities within 200px of each other
        n = len(all_bboxes)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b

        for i in range(n):
            ci = ((all_bboxes[i][0] + all_bboxes[i][2]) / 2,
                  (all_bboxes[i][1] + all_bboxes[i][3]) / 2)
            for j in range(i + 1, n):
                cj = ((all_bboxes[j][0] + all_bboxes[j][2]) / 2,
                      (all_bboxes[j][1] + all_bboxes[j][3]) / 2)
                dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                if dist < 200:
                    union(i, j)

        # Build cluster bounding boxes
        clusters: dict[int, list[tuple]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(all_bboxes[i])

        # Convert clusters to padded ROIs
        padding = 128
        img_w, img_h = image_size
        rois = []
        for bboxes in clusters.values():
            x1 = max(0, min(b[0] for b in bboxes) - padding)
            y1 = max(0, min(b[1] for b in bboxes) - padding)
            x2 = min(img_w, max(b[2] for b in bboxes) + padding)
            y2 = min(img_h, max(b[3] for b in bboxes) + padding)
            rois.append((x1, y1, x2, y2))

        # Merge overlapping ROIs
        return self._merge_overlapping_rois(rois)

    def _merge_overlapping_rois(
        self, rois: list[tuple[float, float, float, float]]
    ) -> list[tuple[float, float, float, float]]:
        """Merge ROIs that overlap with each other."""
        if len(rois) <= 1:
            return rois

        merged = True
        while merged:
            merged = False
            new_rois = []
            used = set()
            for i in range(len(rois)):
                if i in used:
                    continue
                rx1, ry1, rx2, ry2 = rois[i]
                for j in range(i + 1, len(rois)):
                    if j in used:
                        continue
                    ox1, oy1, ox2, oy2 = rois[j]
                    # Check overlap
                    if rx1 <= ox2 and rx2 >= ox1 and ry1 <= oy2 and ry2 >= oy1:
                        rx1 = min(rx1, ox1)
                        ry1 = min(ry1, oy1)
                        rx2 = max(rx2, ox2)
                        ry2 = max(ry2, oy2)
                        used.add(j)
                        merged = True
                new_rois.append((rx1, ry1, rx2, ry2))
                used.add(i)
            rois = new_rois

        return rois

    def _sahi_detect_rois(
        self, image: "Image.Image", rois: list[tuple[float, float, float, float]]
    ) -> list[DetectedEntity]:
        """Run SAHI detection only on specified ROI regions.

        Tiles each ROI into 640x640 chunks and batches all tiles for
        one ONNX call (or sequential PyTorch calls).
        """
        tile_size = 640
        overlap = 32
        stride = tile_size - overlap

        tiles = []
        offsets = []

        for roi in rois:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            for y in range(ry1, ry2, stride):
                for x in range(rx1, rx2, stride):
                    x_end = min(x + tile_size, rx2)
                    y_end = min(y + tile_size, ry2)
                    tile = image.crop((x, y, x_end, y_end))
                    tile_w, tile_h = tile.size
                    if tile.size != (tile_size, tile_size):
                        from PIL import Image as PILImage
                        padded = PILImage.new("RGB", (tile_size, tile_size), (0, 0, 0))
                        padded.paste(tile, (0, 0))
                        tile = padded
                    tiles.append(tile)
                    offsets.append((x, y, tile_w, tile_h))

        if not tiles:
            return []

        all_entities: list[DetectedEntity] = []

        if self.backend == 'onnx' and self.onnx_session:
            # Batch ONNX inference (same approach as _onnx_sahi_detect)
            batch = np.stack([
                np.transpose(np.array(t).astype(np.float32) / 255.0, (2, 0, 1))
                for t in tiles
            ], axis=0)
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: batch})
            raw_output = outputs[0]
            num_classes = len(self.class_names)
            for tile_idx in range(len(tiles)):
                x_off, y_off, tw, th = offsets[tile_idx]
                tile_entities = self._parse_onnx_tile(
                    raw_output, tile_idx, num_classes,
                    scale_x=1.0, scale_y=1.0,
                    x_offset=x_off, y_offset=y_off,
                    clip_w=tw, clip_h=th,
                )
                all_entities.extend(tile_entities)
        else:
            # Sequential PyTorch
            for tile_idx, tile in enumerate(tiles):
                results = self.model(tile, conf=self.confidence_threshold,
                                     imgsz=tile_size, verbose=False)
                x_off, y_off, tw, th = offsets[tile_idx]
                for result in results:
                    if result.boxes is None:
                        continue
                    for box, cls_id, conf in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                        result.boxes.conf.cpu().numpy()
                    ):
                        x1, y1, x2, y2 = box.tolist()
                        class_idx = int(cls_id)
                        class_name = (self.class_names[class_idx]
                                      if class_idx < len(self.class_names)
                                      else f"unknown_{class_idx}")
                        abs_x1 = x1 + x_off
                        abs_y1 = y1 + y_off
                        abs_x2 = min(x2 + x_off, x_off + tw)
                        abs_y2 = min(y2 + y_off, y_off + th)
                        if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                            continue
                        all_entities.append(DetectedEntity(
                            id=self._generate_id(class_name),
                            class_name=class_name,
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                            confidence=float(conf),
                            area=(abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                        ))

        logger.debug("SAHI ROI tiles=%d entities=%d", len(tiles), len(all_entities))
        return all_entities

    def _merge_detections(
        self,
        fast_entities: list[DetectedEntity],
        sahi_entities: list[DetectedEntity],
        rois: list[tuple[float, float, float, float]],
    ) -> list[DetectedEntity]:
        """Merge fast scan + SAHI detections.

        Keep fast entities outside ROIs (reliable at full resolution),
        use SAHI entities inside ROIs (more accurate for small objects).
        """
        merged = []

        # Keep fast entities whose centers are outside all ROI regions
        for e in fast_entities:
            cx, cy = e.center
            in_roi = any(
                rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                for rx1, ry1, rx2, ry2 in rois
            )
            if not in_roi:
                merged.append(e)

        # Add all SAHI entities from ROI regions
        merged.extend(sahi_entities)

        return merged

    def _pytorch_detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
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
            return self._sahi_detect(image)

        # Standard inference for small images
        results = self.model(image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False)
        return self._parse_yolo_results(results)

    def _sahi_detect(self, image: "Image.Image") -> list[DetectedEntity]:
        """SAHI sliced inference: tile the image into overlapping 640x640 chunks.

        On Retina displays (3024x1672), standard inference resizes to imgsz=1280,
        shrinking sheep to ~21px — below reliable detection. SAHI runs each tile
        at native 640x640 (the model's training resolution), then offsets coordinates
        back to the original image space. Existing NMS in detect() deduplicates.

        ~24 tiles * ~40ms/tile ≈ 960ms total — acceptable vs 1-3s LLM API call.
        """
        width, height = image.size
        tile_size = 640
        overlap = 32  # 5% overlap — NMS handles boundary duplicates
        stride = tile_size - overlap
        all_entities = []

        for y_start in range(0, height, stride):
            for x_start in range(0, width, stride):
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                tile = image.crop((x_start, y_start, x_end, y_end))
                results = self.model(tile, conf=self.confidence_threshold,
                                     imgsz=tile_size, verbose=False)

                for result in results:
                    if result.boxes is None:
                        continue
                    for box, cls_id, conf in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                        result.boxes.conf.cpu().numpy()
                    ):
                        x1, y1, x2, y2 = box.tolist()
                        class_idx = int(cls_id)
                        class_name = (self.class_names[class_idx]
                                      if class_idx < len(self.class_names)
                                      else f"unknown_{class_idx}")

                        # Offset coordinates to original image space
                        abs_x1 = x1 + x_start
                        abs_y1 = y1 + y_start
                        abs_x2 = x2 + x_start
                        abs_y2 = y2 + y_start

                        all_entities.append(DetectedEntity(
                            id=self._generate_id(class_name),
                            class_name=class_name,
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                            confidence=float(conf),
                            area=(abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                        ))

        # Sort by class name, then by confidence (highest first)
        all_entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return all_entities

    def _onnx_sahi_detect(self, image: "Image.Image") -> list[DetectedEntity]:
        """ONNX batched SAHI: tile the image, batch all tiles, run one ONNX call.

        Same tiling logic as _sahi_detect() but batches all tiles into a single
        ONNX Runtime inference call for ~3-5x speedup over sequential PyTorch.
        """
        width, height = image.size
        tile_size = 640
        overlap = 32
        stride = tile_size - overlap

        # 1. Collect all tiles and their offsets
        tiles = []
        offsets = []
        for y_start in range(0, height, stride):
            for x_start in range(0, width, stride):
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)
                tile = image.crop((x_start, y_start, x_end, y_end))
                # Pad tile to tile_size x tile_size if it's at the edge
                if tile.size != (tile_size, tile_size):
                    from PIL import Image as PILImage
                    padded = PILImage.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0))
                    tile = padded
                tiles.append(tile)
                offsets.append((x_start, y_start, x_end - x_start, y_end - y_start))

        if not tiles:
            return []

        # 2. Pre-process all tiles into a single batch
        batch = np.stack([
            np.transpose(np.array(t).astype(np.float32) / 255.0, (2, 0, 1))
            for t in tiles
        ], axis=0)  # (N, 3, 640, 640)

        # 3. Single ONNX inference call
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: batch})
        raw_output = outputs[0]

        logger.debug("ONNX SAHI batch shape: %s, tiles: %d", raw_output.shape, len(tiles))

        # 4. Parse results per tile and offset coordinates
        all_entities = []
        num_classes = len(self.class_names)

        for tile_idx in range(len(tiles)):
            x_start, y_start, tile_w, tile_h = offsets[tile_idx]
            tile_entities = self._parse_onnx_tile(
                raw_output, tile_idx, num_classes,
                scale_x=1.0, scale_y=1.0,  # Tiles are already at native 640x640
                x_offset=x_start, y_offset=y_start,
                clip_w=tile_w, clip_h=tile_h,
            )
            all_entities.extend(tile_entities)

        all_entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return all_entities

    def _parse_onnx_tile(
        self, raw_output: np.ndarray, tile_idx: int, num_classes: int,
        scale_x: float, scale_y: float,
        x_offset: float = 0, y_offset: float = 0,
        clip_w: float = 640, clip_h: float = 640,
    ) -> list[DetectedEntity]:
        """Parse ONNX output for a single tile from a batched result.

        Handles both output formats:
        - Post-NMS: (batch, num_detections, 6) → [x1, y1, x2, y2, conf, cls]
        - Raw: (batch, 4+num_classes, num_boxes) → needs argmax + threshold
        """
        entities = []

        if len(raw_output.shape) == 3 and raw_output.shape[2] == 6:
            # Post-NMS format: (batch, num_detections, 6)
            predictions = raw_output[tile_idx]
        elif len(raw_output.shape) == 3 and raw_output.shape[1] == (4 + num_classes):
            # Raw format: (batch, 4+num_classes, num_boxes)
            preds_raw = raw_output[tile_idx].T  # (num_boxes, 4+num_classes)
            boxes = preds_raw[:, :4]
            class_scores = preds_raw[:, 4:]
            best_cls = np.argmax(class_scores, axis=1)
            best_conf = np.max(class_scores, axis=1)
            # Use lowest per-class threshold for initial bulk filter
            min_thresh = min(self.class_thresholds.values()) if self.class_thresholds else self.confidence_threshold
            mask = best_conf >= min_thresh
            boxes, best_cls, best_conf = boxes[mask], best_cls[mask], best_conf[mask]
            predictions = []
            for box, cls_id, conf in zip(boxes, best_cls, best_conf):
                xc, yc, w, h = box
                predictions.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2, conf, cls_id])
            predictions = np.array(predictions) if predictions else np.array([]).reshape(0, 6)
        else:
            return []

        for pred in predictions:
            x1, y1, x2, y2, confidence, class_id = pred
            class_idx = int(class_id)
            class_name = (self.class_names[class_idx]
                          if class_idx < len(self.class_names)
                          else f"unknown_{class_idx}")
            if confidence < self._get_threshold(class_name):
                continue

            # Scale and offset to original image space
            abs_x1 = x1 * scale_x + x_offset
            abs_y1 = y1 * scale_y + y_offset
            abs_x2 = x2 * scale_x + x_offset
            abs_y2 = y2 * scale_y + y_offset

            # Clip to tile's actual region (handles edge padding)
            abs_x2 = min(abs_x2, x_offset + clip_w)
            abs_y2 = min(abs_y2, y_offset + clip_h)

            if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                continue

            entities.append(DetectedEntity(
                id=self._generate_id(class_name),
                class_name=class_name,
                bbox=(float(abs_x1), float(abs_y1), float(abs_x2), float(abs_y2)),
                center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                confidence=float(confidence),
                area=float((abs_x2 - abs_x1) * (abs_y2 - abs_y1)),
            ))

        return entities

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
                boxes.conf.cpu().numpy()
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
                    area=area
                )
                entities.append(entity)

        entities.sort(key=lambda e: (e.class_name, -e.confidence))
        return entities

    def _onnx_detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
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

            # Debug: show confidence distribution
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
            min_thresh = min(self.class_thresholds.values()) if self.class_thresholds else self.confidence_threshold
            mask = best_confidence >= min_thresh
            boxes = boxes[mask]
            best_class_idx = best_class_idx[mask]
            best_confidence = best_confidence[mask]

            logger.debug("%d boxes after confidence filter (%.2f)", len(boxes), min_thresh)

            # Convert from x_center, y_center, w, h to x1, y1, x2, y2
            predictions = []
            for box, cls_id, conf in zip(boxes, best_class_idx, best_confidence):
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
            class_name = (self.class_names[class_idx]
                          if class_idx < len(self.class_names)
                          else f"unknown_{class_idx}")
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
                area=float(area)
            )
            entities.append(entity)

        # Sort by class name, then by confidence (highest first)
        entities.sort(key=lambda e: (e.class_name, -e.confidence))

        logger.debug("Final entity count: %d", len(entities))

        return entities

    def _nms(self, entities: list[DetectedEntity], iou_threshold: float = 0.5) -> list[DetectedEntity]:
        """Simple non-maximum suppression."""
        if not entities:
            return []

        # Sort by confidence (highest first)
        entities = sorted(entities, key=lambda e: -e.confidence)

        keep = []
        while entities:
            best = entities.pop(0)
            keep.append(best)

            # Filter out overlapping boxes of the same class
            entities = [
                e for e in entities
                if e.class_name != best.class_name or self._iou(best.bbox, e.bbox) < iou_threshold
            ]

        return keep

    def _iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate intersection over union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _mock_detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
        """Generate mock detections for testing.

        Creates plausible detections based on typical Dark Age game state.
        """
        from PIL import Image
        import random

        # Get image dimensions
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
            width, height = image.size
        elif hasattr(screenshot, 'size'):
            width, height = screenshot.size
        else:
            width, height = 1920, 1080  # Default fallback

        # Set random seed for reproducibility within same session
        random.seed(42)

        entities = []

        # Town Center - usually center of screen
        tc_x = width * 0.5 + random.uniform(-100, 100)
        tc_y = height * 0.5 + random.uniform(-50, 50)
        entities.append(DetectedEntity(
            id=self._generate_id("town_center"),
            class_name="town_center",
            bbox=(tc_x - 80, tc_y - 60, tc_x + 80, tc_y + 60),
            center=(tc_x, tc_y),
            confidence=0.95,
            area=160 * 120
        ))

        # Sheep - typically 2-4 near TC at game start
        for i in range(random.randint(2, 4)):
            sheep_x = tc_x + random.uniform(-200, 200)
            sheep_y = tc_y + random.uniform(-150, 150)
            entities.append(DetectedEntity(
                id=self._generate_id("sheep"),
                class_name="sheep",
                bbox=(sheep_x - 15, sheep_y - 10, sheep_x + 15, sheep_y + 10),
                center=(sheep_x, sheep_y),
                confidence=random.uniform(0.7, 0.95),
                area=30 * 20
            ))

        # Villagers - 3 starting villagers
        for i in range(3):
            vill_x = tc_x + random.uniform(-150, 150)
            vill_y = tc_y + random.uniform(-100, 100)
            entities.append(DetectedEntity(
                id=self._generate_id("villager"),
                class_name="villager",
                bbox=(vill_x - 12, vill_y - 20, vill_x + 12, vill_y + 5),
                center=(vill_x, vill_y),
                confidence=random.uniform(0.75, 0.92),
                area=24 * 25
            ))

        # Scout - usually exploring
        scout_x = random.uniform(100, width - 100)
        scout_y = random.uniform(100, height - 100)
        entities.append(DetectedEntity(
            id=self._generate_id("scout"),
            class_name="scout",
            bbox=(scout_x - 15, scout_y - 18, scout_x + 15, scout_y + 8),
            center=(scout_x, scout_y),
            confidence=0.88,
            area=30 * 26
        ))

        # Sort by class name, then by confidence
        entities.sort(key=lambda e: (e.class_name, -e.confidence))

        return entities

    def detect_to_dict_list(self, screenshot: Union[bytes, "Image.Image"]) -> list[dict]:
        """Detect and return as list of dictionaries.

        Convenience method for LLM context building.
        """
        return [e.to_dict() for e in self.detect(screenshot)]

    def find_entity_by_id(
        self,
        entities: list[DetectedEntity],
        target_id: str
    ) -> Optional[DetectedEntity]:
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
        self,
        entities: list[DetectedEntity],
        class_name: str
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
        class_filter: Optional[str] = None
    ) -> Optional[DetectedEntity]:
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
_instance: Optional[EntityDetector] = None

def get_detector(
    model_path: Optional[str] = None,
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
