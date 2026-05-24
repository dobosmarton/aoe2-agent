"""Unit tests for the pure SAHI helpers in `detection.inference.sahi`.

`compute_sahi_rois`, `merge_overlapping_rois`, `merge_detections`, and
`generate_tiles` don't depend on a YOLO model — they just transform geometry.
The model-using helpers (`sahi_detect`, `onnx_sahi_detect`,
`sahi_detect_rois`, `parse_onnx_tile`) need real weights or ONNX outputs to
exercise meaningfully and are intentionally skipped here.
"""

from __future__ import annotations

from detection.inference.detector import DetectedEntity
from detection.inference.sahi import (
    SAHI_OVERLAP,
    SAHI_TILE_SIZE,
    compute_sahi_rois,
    generate_tiles,
    merge_detections,
    merge_overlapping_rois,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ent(class_name: str, bbox: tuple, confidence: float = 0.9) -> DetectedEntity:
    return DetectedEntity(
        id=f"{class_name}_x",
        class_name=class_name,
        bbox=bbox,
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# merge_overlapping_rois
# ---------------------------------------------------------------------------


def test_merge_rois_empty_returns_empty():
    assert merge_overlapping_rois([]) == []


def test_merge_rois_single_passes_through():
    rois = [(0, 0, 100, 100)]
    assert merge_overlapping_rois(rois) == rois


def test_merge_rois_disjoint_unchanged():
    rois = [(0, 0, 50, 50), (200, 200, 250, 250)]
    out = merge_overlapping_rois(rois)
    assert len(out) == 2


def test_merge_rois_two_overlapping_become_one():
    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    out = merge_overlapping_rois([a, b])
    assert len(out) == 1
    assert out[0] == (0, 0, 150, 150)


def test_merge_rois_chain_collapses():
    """A → B and B → C should all merge into one ROI in subsequent passes."""
    a = (0, 0, 60, 60)
    b = (50, 50, 110, 110)
    c = (100, 100, 160, 160)
    out = merge_overlapping_rois([a, b, c])
    assert len(out) == 1
    assert out[0] == (0, 0, 160, 160)


# ---------------------------------------------------------------------------
# compute_sahi_rois
# ---------------------------------------------------------------------------


def test_compute_rois_no_entities_returns_empty():
    assert compute_sahi_rois([], [], (1920, 1080)) == []


def test_compute_rois_pads_around_single_entity():
    """A single entity becomes one ROI with 128px padding (clipped to image bounds)."""
    e = _ent("sheep", (500, 500, 520, 520))
    out = compute_sahi_rois([e], [], (1920, 1080))
    assert len(out) == 1
    x1, y1, x2, y2 = out[0]
    assert x1 == 500 - 128
    assert y1 == 500 - 128
    assert x2 == 520 + 128
    assert y2 == 520 + 128


def test_compute_rois_clips_to_image_bounds():
    """Entity near the corner — ROI must not extend beyond (0, 0) or image size."""
    e = _ent("sheep", (10, 10, 30, 30))
    out = compute_sahi_rois([e], [], (1920, 1080))
    x1, y1, _, _ = out[0]
    assert x1 == 0
    assert y1 == 0


def test_compute_rois_clusters_nearby_entities():
    """Two entities within 200px → single shared ROI, not two."""
    a = _ent("sheep", (500, 500, 520, 520))
    b = _ent("sheep", (600, 600, 620, 620))
    out = compute_sahi_rois([a, b], [], (1920, 1080))
    assert len(out) == 1


def test_compute_rois_separates_distant_clusters():
    """Two entities >200px apart → two distinct ROIs (assuming padding doesn't make them overlap)."""
    a = _ent("sheep", (100, 100, 110, 110))
    b = _ent("sheep", (1500, 1000, 1510, 1010))
    out = compute_sahi_rois([a, b], [], (1920, 1080))
    assert len(out) == 2


def test_compute_rois_includes_disappeared_previous_entities():
    """A previous entity not matching any current one should still get tiled — it may have moved."""
    fast = [_ent("sheep", (500, 500, 520, 520))]
    prev = [_ent("sheep", (1500, 1000, 1520, 1020))]
    out = compute_sahi_rois(fast, prev, (1920, 1080))
    assert len(out) == 2


def test_compute_rois_skips_previous_still_visible():
    """A previous entity overlapping a current one shouldn't add a redundant ROI."""
    current = _ent("sheep", (500, 500, 520, 520))
    prev = _ent("sheep", (502, 502, 522, 522))
    out = compute_sahi_rois([current], [prev], (1920, 1080))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# merge_detections
# ---------------------------------------------------------------------------


def test_merge_detections_keeps_fast_outside_rois():
    fast = _ent("tree", (1000, 100, 1020, 120))
    rois: list[tuple[float, float, float, float]] = [(0, 0, 200, 200)]
    out = merge_detections([fast], [], rois)
    assert out == [fast]


def test_merge_detections_drops_fast_inside_rois():
    """A fast-pass entity whose center sits inside an ROI is replaced by the SAHI version."""
    inside = _ent("sheep", (50, 50, 70, 70))
    rois: list[tuple[float, float, float, float]] = [(0, 0, 200, 200)]
    out = merge_detections([inside], [], rois)
    assert out == []


def test_merge_detections_appends_all_sahi_entities():
    sahi = _ent("sheep", (50, 50, 70, 70))
    rois: list[tuple[float, float, float, float]] = [(0, 0, 200, 200)]
    out = merge_detections([], [sahi], rois)
    assert out == [sahi]


def test_merge_detections_combines_outside_fast_and_sahi():
    """Typical adaptive case: outside-ROI fast detections + SAHI inside ROIs."""
    far = _ent("tree", (1500, 100, 1520, 120))
    inside_fast = _ent("sheep", (100, 100, 120, 120))  # dropped, replaced by sahi
    sahi = _ent("sheep", (102, 102, 122, 122))
    rois: list[tuple[float, float, float, float]] = [(0, 0, 200, 200)]
    out = merge_detections([far, inside_fast], [sahi], rois)
    assert {e.id for e in out} == {far.id, sahi.id}


# ---------------------------------------------------------------------------
# generate_tiles
# ---------------------------------------------------------------------------


def test_generate_tiles_full_image_when_no_rois():
    """A 1280x720 image at default tile size 640 with overlap 32 → multiple tiles."""
    image = Image.new("RGB", (1280, 720))
    tiles, offsets = generate_tiles(image)
    assert len(tiles) == len(offsets)
    assert len(tiles) > 1


def test_generate_tiles_pads_to_tile_size():
    """Edge tiles smaller than 640x640 get padded to 640x640 for batched inference."""
    image = Image.new("RGB", (700, 700))
    tiles, _ = generate_tiles(image)
    for tile in tiles:
        assert tile.size == (SAHI_TILE_SIZE, SAHI_TILE_SIZE)


def test_generate_tiles_offset_records_unpadded_size():
    """Offsets should report the actual cropped tile size, not the padded size."""
    image = Image.new("RGB", (700, 700))
    _, offsets = generate_tiles(image)
    # Last tile in each row is the trimmed one — its tile_w/tile_h < SAHI_TILE_SIZE
    smaller = [o for o in offsets if o[2] < SAHI_TILE_SIZE or o[3] < SAHI_TILE_SIZE]
    assert smaller, "expected at least one edge-padded tile in a 700x700 image"


def test_generate_tiles_respects_rois():
    """When ROIs are given, tiles only cover those regions, not the whole image."""
    image = Image.new("RGB", (2000, 2000))
    rois: list[tuple[float, float, float, float]] = [(0, 0, 640, 640)]
    tiles_with_rois, _ = generate_tiles(image, rois=rois)
    tiles_full, _ = generate_tiles(image)
    # ROI tiling never exceeds full-image tiling, and skipping the rest of a
    # 2000x2000 image cuts the tile count substantially.
    assert len(tiles_with_rois) < len(tiles_full)
    assert len(tiles_with_rois) <= 4  # 640x640 ROI at stride 608 → at most 2x2 tiles


def test_generate_tiles_empty_image_smaller_than_tile():
    """Image smaller than a single tile still produces one (padded) tile."""
    image = Image.new("RGB", (200, 200))
    tiles, offsets = generate_tiles(image)
    assert len(tiles) == 1
    assert tiles[0].size == (SAHI_TILE_SIZE, SAHI_TILE_SIZE)
    # Offset records the unpadded crop size
    assert offsets[0][2] == 200
    assert offsets[0][3] == 200


def test_sahi_constants_make_sense():
    """Stride must be positive and less than the tile size — guards against config typos."""
    assert SAHI_OVERLAP > 0
    assert SAHI_OVERLAP < SAHI_TILE_SIZE
    assert SAHI_TILE_SIZE - SAHI_OVERLAP > 0
