"""Direct unit tests for `detection.inference.postprocess`.

These functions are exercised indirectly through the detector smoke tests, but
direct tests pin the contract at the function boundary — useful when refactoring
the post-processing logic without touching `EntityDetector`.
"""

from __future__ import annotations

from detection.inference.detector import DetectedEntity
from detection.inference.postprocess import iou, nms

# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = (0, 0, 10, 10)
    b = (100, 100, 110, 110)
    assert iou(a, b) == 0.0


def test_iou_touching_boxes_is_zero():
    """Boxes that share an edge but don't overlap have zero IoU."""
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    """Two 10x10 boxes overlapping by 5x5 → intersection=25, union=175 → 1/7."""
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert iou(a, b) == 25 / (100 + 100 - 25)


def test_iou_one_inside_other():
    """Inner box fully contained → IoU = inner_area / outer_area."""
    outer = (0, 0, 10, 10)
    inner = (2, 2, 4, 4)
    assert iou(outer, inner) == 4 / 100


def test_iou_zero_area_box_is_zero():
    """Degenerate boxes (zero area) have IoU 0 — guards against div-by-zero."""
    zero = (5, 5, 5, 5)
    box = (0, 0, 10, 10)
    assert iou(zero, box) == 0.0


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def _ent(class_name: str, bbox: tuple, confidence: float, eid: str = "x") -> DetectedEntity:
    return DetectedEntity(
        id=eid,
        class_name=class_name,
        bbox=bbox,
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        confidence=confidence,
    )


def test_nms_empty_input_returns_empty():
    assert nms([]) == []


def test_nms_single_entity_passes_through():
    e = _ent("sheep", (0, 0, 10, 10), 0.9)
    assert nms([e]) == [e]


def test_nms_suppresses_lower_confidence_overlap_same_class():
    """Two highly-overlapping same-class boxes → only the higher-confidence one survives."""
    high = _ent("sheep", (0, 0, 10, 10), 0.95, eid="high")
    low = _ent("sheep", (1, 1, 11, 11), 0.50, eid="low")
    out = nms([high, low], iou_threshold=0.5)
    assert out == [high]


def test_nms_keeps_overlap_across_different_classes():
    """A villager standing on a sheep is fine — different classes don't suppress each other."""
    a = _ent("sheep", (0, 0, 10, 10), 0.9, eid="sheep")
    b = _ent("villager", (1, 1, 11, 11), 0.85, eid="vill")
    out = nms([a, b], iou_threshold=0.5)
    assert {e.id for e in out} == {"sheep", "vill"}


def test_nms_keeps_non_overlapping_same_class():
    a = _ent("sheep", (0, 0, 10, 10), 0.9, eid="a")
    b = _ent("sheep", (100, 100, 110, 110), 0.85, eid="b")
    out = nms([a, b])
    assert {e.id for e in out} == {"a", "b"}


def test_nms_threshold_governs_suppression():
    """With a high enough threshold, even overlapping boxes survive."""
    a = _ent("sheep", (0, 0, 10, 10), 0.9, eid="a")
    b = _ent("sheep", (5, 5, 15, 15), 0.85, eid="b")  # IoU ≈ 0.143
    survives = nms([a, b], iou_threshold=0.95)
    suppressed = nms([a, b], iou_threshold=0.10)
    assert len(survives) == 2
    assert len(suppressed) == 1


def test_nms_processes_in_confidence_order():
    """Result ordering reflects highest-confidence-first iteration."""
    low = _ent("sheep", (0, 0, 10, 10), 0.50, eid="low")
    high = _ent("sheep", (100, 100, 110, 110), 0.95, eid="high")
    mid = _ent("sheep", (200, 200, 210, 210), 0.70, eid="mid")
    out = nms([low, high, mid])
    assert [e.id for e in out] == ["high", "mid", "low"]
