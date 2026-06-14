"""SAHI sliced inference for high-resolution screenshots.

`detect()` at native imgsz=1280 shrinks small entities (~21px sheep) below
reliable detection. SAHI tiles the image into overlapping 640x640 chunks
(the model's training resolution), runs inference per tile, and offsets
coordinates back to the source image. Subsequent NMS deduplicates the
boundary detections.

Two backends share the tiling logic but differ in inference call shape:

  - PyTorch: `sahi_detect` runs each tile sequentially through ultralytics.
  - ONNX: `onnx_sahi_detect` batches all tiles into one ONNXRuntime call
    (~3-5x faster than sequential).

The adaptive variant (`compute_sahi_rois` + `sahi_detect_rois` +
`merge_detections`) tiles only ROI regions around clustered entities,
cutting tile counts from ~18 to ~3-8 when the detection set is sparse.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, cast

import numpy as np

from ._ultralytics_results import yolo_boxes_to_lists
from .onnx_layout import UnknownOnnxLayoutError, decode_example
from .postprocess import iou

if TYPE_CHECKING:
    from PIL import Image

    from .detector import DetectedEntity, EntityDetector


logger = logging.getLogger(__name__)


SAHI_TILE_SIZE = 640
SAHI_OVERLAP = 32


def compute_sahi_rois(
    fast_entities: list[DetectedEntity],
    previous_entities: list[DetectedEntity],
    image_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """Compute ROI regions for targeted SAHI based on entity clusters.

    Groups entities within 200px into clusters, adds 128px padding,
    and merges overlapping ROIs. Also includes regions where previous
    entities disappeared (may have moved just beyond fast-pass range).
    """
    all_bboxes = [e.bbox for e in fast_entities]

    for prev in previous_entities:
        if not any(iou(prev.bbox, f.bbox) > 0.3 for f in fast_entities):
            all_bboxes.append(prev.bbox)

    if not all_bboxes:
        return []

    n = len(all_bboxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i in range(n):
        ci = (
            (all_bboxes[i][0] + all_bboxes[i][2]) / 2,
            (all_bboxes[i][1] + all_bboxes[i][3]) / 2,
        )
        for j in range(i + 1, n):
            cj = (
                (all_bboxes[j][0] + all_bboxes[j][2]) / 2,
                (all_bboxes[j][1] + all_bboxes[j][3]) / 2,
            )
            dx = float(ci[0] - cj[0])
            dy = float(ci[1] - cj[1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 200:
                union(i, j)

    clusters: dict[int, list[tuple]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(all_bboxes[i])

    padding = 128
    img_w, img_h = image_size
    rois = []
    for bboxes in clusters.values():
        x1 = max(0, min(b[0] for b in bboxes) - padding)
        y1 = max(0, min(b[1] for b in bboxes) - padding)
        x2 = min(img_w, max(b[2] for b in bboxes) + padding)
        y2 = min(img_h, max(b[3] for b in bboxes) + padding)
        rois.append((x1, y1, x2, y2))

    return merge_overlapping_rois(rois)


def merge_overlapping_rois(
    rois: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Merge ROIs that overlap with each other."""
    if len(rois) <= 1:
        return rois

    merged_flag = True
    while merged_flag:
        merged_flag = False
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
                if rx1 <= ox2 and rx2 >= ox1 and ry1 <= oy2 and ry2 >= oy1:
                    rx1 = min(rx1, ox1)
                    ry1 = min(ry1, oy1)
                    rx2 = max(rx2, ox2)
                    ry2 = max(ry2, oy2)
                    used.add(j)
                    merged_flag = True
            new_rois.append((rx1, ry1, rx2, ry2))
            used.add(i)
        rois = new_rois

    return rois


def generate_tiles(
    image: Image.Image,
    rois: list[tuple[float, float, float, float]] | None = None,
) -> tuple[list, list[tuple[int, int, int, int]]]:
    """Generate padded tiles and their (x, y, tile_w, tile_h) offsets.

    If `rois` is given, only tiles within those regions are produced.
    Otherwise the whole image is tiled.
    """
    from PIL import Image as PILImage

    tile_size = SAHI_TILE_SIZE
    stride = tile_size - SAHI_OVERLAP

    if rois:
        regions = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rois]
    else:
        width, height = image.size
        regions = [(0, 0, width, height)]

    tiles: list = []
    offsets: list[tuple[int, int, int, int]] = []

    for rx1, ry1, rx2, ry2 in regions:
        for y in range(ry1, ry2, stride):
            for x in range(rx1, rx2, stride):
                x_end = min(x + tile_size, rx2)
                y_end = min(y + tile_size, ry2)
                tile = image.crop((x, y, x_end, y_end))
                tile_w, tile_h = tile.size
                if tile.size != (tile_size, tile_size):
                    padded = PILImage.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0))
                    tile = padded
                tiles.append(tile)
                offsets.append((x, y, tile_w, tile_h))

    return tiles, offsets


def sahi_detect_rois(
    det: EntityDetector,
    image: Image.Image,
    rois: list[tuple[float, float, float, float]],
) -> list[DetectedEntity]:
    """Run SAHI detection only on specified ROI regions.

    Tiles each ROI into 640x640 chunks and batches all tiles for
    one ONNX call (or sequential PyTorch calls).
    """
    from .detector import DetectedEntity

    tile_size = SAHI_TILE_SIZE
    tiles, offsets = generate_tiles(image, rois=rois)

    if not tiles:
        return []

    all_entities: list[DetectedEntity] = []

    if det.backend == "onnx" and det.onnx_session:
        batch = np.stack(
            [np.transpose(np.array(t).astype(np.float32) / 255.0, (2, 0, 1)) for t in tiles],
            axis=0,
        )
        input_name = cast("str", det.onnx_session.get_inputs()[0].name)
        outputs = cast("list[np.ndarray]", det.onnx_session.run(None, {input_name: batch}))
        raw_output = outputs[0]
        for tile_idx in range(len(tiles)):
            x_off, y_off, tw, th = offsets[tile_idx]
            tile_entities = parse_onnx_tile(
                det,
                raw_output,
                tile_idx,
                scale_x=1.0,
                scale_y=1.0,
                x_offset=x_off,
                y_offset=y_off,
                clip_w=tw,
                clip_h=th,
            )
            all_entities.extend(tile_entities)
    else:
        if det.model is None:
            raise RuntimeError("det.model not loaded")
        for tile_idx, tile in enumerate(tiles):
            results = cast(
                "list[object]",
                det.model(  # pyright: ignore[reportAny]
                    tile, conf=det.confidence_threshold, imgsz=tile_size, verbose=False
                ),
            )
            x_off, y_off, tw, th = offsets[tile_idx]
            for result in results:
                boxes_attr: object | None = getattr(result, "boxes", None)
                if boxes_attr is None:
                    continue
                bboxes, classes, confidences = yolo_boxes_to_lists(boxes_attr)
                for box, cls_id, conf in zip(bboxes, classes, confidences, strict=True):
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    class_idx = int(cls_id)
                    class_name = (
                        det.class_names[class_idx]
                        if class_idx < len(det.class_names)
                        else f"unknown_{class_idx}"
                    )
                    abs_x1 = x1 + x_off
                    abs_y1 = y1 + y_off
                    abs_x2 = min(x2 + x_off, x_off + tw)
                    abs_y2 = min(y2 + y_off, y_off + th)
                    if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                        continue
                    all_entities.append(
                        DetectedEntity(
                            id=det._generate_id(class_name),
                            class_name=class_name,
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                            confidence=conf,
                            area=(abs_x2 - abs_x1) * (abs_y2 - abs_y1),
                        )
                    )

    logger.debug("SAHI ROI tiles=%d entities=%d", len(tiles), len(all_entities))
    return all_entities


def merge_detections(
    fast_entities: list[DetectedEntity],
    sahi_entities: list[DetectedEntity],
    rois: list[tuple[float, float, float, float]],
) -> list[DetectedEntity]:
    """Merge fast scan + SAHI detections.

    Keep fast entities outside ROIs (reliable at full resolution),
    use SAHI entities inside ROIs (more accurate for small objects).
    """
    merged = []

    for e in fast_entities:
        cx, cy = e.center
        in_roi = any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in rois)
        if not in_roi:
            merged.append(e)

    merged.extend(sahi_entities)

    return merged


def sahi_detect(det: EntityDetector, image: Image.Image) -> list[DetectedEntity]:
    """SAHI sliced inference: tile the image into overlapping 640x640 chunks.

    On Retina displays (3024x1672), standard inference resizes to imgsz=1280,
    shrinking sheep to ~21px — below reliable detection. SAHI runs each tile
    at native 640x640 (the model's training resolution), then offsets coordinates
    back to the original image space. Existing NMS in detect() deduplicates.

    ~24 tiles * ~40ms/tile ≈ 960ms total — acceptable vs 1-3s LLM API call.
    """
    from .detector import DetectedEntity

    width, height = image.size
    tile_size = 640
    overlap = 32
    stride = tile_size - overlap
    all_entities = []
    if det.model is None:
        raise RuntimeError("det.model not loaded")

    for y_start in range(0, height, stride):
        for x_start in range(0, width, stride):
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)

            tile = image.crop((x_start, y_start, x_end, y_end))
            results = cast(
                "list[object]",
                det.model(  # pyright: ignore[reportAny]
                    tile, conf=det.confidence_threshold, imgsz=tile_size, verbose=False
                ),
            )

            for result in results:
                boxes_attr: object | None = getattr(result, "boxes", None)
                if boxes_attr is None:
                    continue
                bboxes, classes, confidences = yolo_boxes_to_lists(boxes_attr)
                for box, cls_id, conf in zip(bboxes, classes, confidences, strict=True):
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    class_idx = int(cls_id)
                    class_name = (
                        det.class_names[class_idx]
                        if class_idx < len(det.class_names)
                        else f"unknown_{class_idx}"
                    )

                    abs_x1 = x1 + x_start
                    abs_y1 = y1 + y_start
                    abs_x2 = x2 + x_start
                    abs_y2 = y2 + y_start

                    all_entities.append(
                        DetectedEntity(
                            id=det._generate_id(class_name),
                            class_name=class_name,
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                            confidence=conf,
                            area=(abs_x2 - abs_x1) * (abs_y2 - abs_y1),
                        )
                    )

    all_entities.sort(key=lambda e: (e.class_name, -e.confidence))
    return all_entities


def onnx_sahi_detect(det: EntityDetector, image: Image.Image) -> list[DetectedEntity]:
    """ONNX batched SAHI: tile the image, batch all tiles, run one ONNX call.

    Same tiling logic as `sahi_detect()` but batches all tiles into a single
    ONNX Runtime inference call for ~3-5x speedup over sequential PyTorch.
    """
    tiles, offsets = generate_tiles(image)

    if not tiles:
        return []

    batch = np.stack(
        [np.transpose(np.array(t).astype(np.float32) / 255.0, (2, 0, 1)) for t in tiles], axis=0
    )

    if det.onnx_session is None:
        raise RuntimeError("det.onnx_session not initialised")
    input_name = cast("str", det.onnx_session.get_inputs()[0].name)
    outputs = cast("list[np.ndarray]", det.onnx_session.run(None, {input_name: batch}))
    raw_output = outputs[0]

    logger.debug("ONNX SAHI batch shape: %s, tiles: %d", raw_output.shape, len(tiles))

    all_entities = []

    for tile_idx in range(len(tiles)):
        x_start, y_start, tile_w, tile_h = offsets[tile_idx]
        tile_entities = parse_onnx_tile(
            det,
            raw_output,
            tile_idx,
            scale_x=1.0,
            scale_y=1.0,
            x_offset=x_start,
            y_offset=y_start,
            clip_w=tile_w,
            clip_h=tile_h,
        )
        all_entities.extend(tile_entities)

    all_entities.sort(key=lambda e: (e.class_name, -e.confidence))
    return all_entities


def parse_onnx_tile(
    det: EntityDetector,
    raw_output: np.ndarray,
    tile_idx: int,
    scale_x: float,
    scale_y: float,
    x_offset: float = 0,
    y_offset: float = 0,
    clip_w: float = 640,
    clip_h: float = 640,
) -> list[DetectedEntity]:
    """Parse ONNX output for a single tile from a batched result.

    Decoding is shared with the single-image path via `decode_example`; this
    function only adds the per-tile coordinate scale, offset, and clip.
    """
    from .detector import DetectedEntity

    min_confidence = (
        min(det.class_thresholds.values()) if det.class_thresholds else det.confidence_threshold
    )
    try:
        rows = decode_example(cast("np.ndarray", raw_output[tile_idx]), min_confidence)
    except UnknownOnnxLayoutError:
        return []

    entities: list[DetectedEntity] = []
    for row in rows:
        class_name = (
            det.class_names[row.class_id]
            if row.class_id < len(det.class_names)
            else f"unknown_{row.class_id}"
        )
        if row.confidence < det._get_threshold(class_name):
            continue

        abs_x1 = row.x1 * scale_x + x_offset
        abs_y1 = row.y1 * scale_y + y_offset
        abs_x2 = min(row.x2 * scale_x + x_offset, x_offset + clip_w)
        abs_y2 = min(row.y2 * scale_y + y_offset, y_offset + clip_h)

        if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
            continue

        entities.append(
            DetectedEntity(
                id=det._generate_id(class_name),
                class_name=class_name,
                bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                center=((abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2),
                confidence=row.confidence,
                area=(abs_x2 - abs_x1) * (abs_y2 - abs_y1),
            )
        )

    return entities
