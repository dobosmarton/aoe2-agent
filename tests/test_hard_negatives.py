"""Unit tests for cavalry-confusion hard-negative mining.

Pure tests over hand-built `TriageItem`s — no model, no `ultralytics`.
"""

from __future__ import annotations

from detection.labeling.active_learning import DetectionRecord, TriageItem
from detection.labeling.hard_negatives import (
    CAVALRY_CONFUSION_CLASSES,
    find_confusable_detections,
)


def _detection(class_name: str, confidence: float) -> DetectionRecord:
    return DetectionRecord(class_name=class_name, confidence=confidence, bbox=(0.0, 0.0, 1.0, 1.0))


def _item(path: str, detections: tuple[DetectionRecord, ...]) -> TriageItem:
    return TriageItem(
        path=path,
        name=path,
        score=0,
        n_detections=len(detections),
        n_uncertain=0,
        n_low=0,
        detections=detections,
    )


def test_keeps_only_low_confidence_focus_classes() -> None:
    item = _item(
        "a.jpg",
        (
            _detection("knight_line", 0.30),  # confusable + low → hit
            _detection("knight_line", 0.95),  # confusable but confident → skip
            _detection("villager", 0.10),  # low but not confusable → skip
        ),
    )

    hits = find_confusable_detections([item], max_confidence=0.5)

    assert len(hits) == 1
    assert hits[0].detection.class_name == "knight_line"
    assert hits[0].image == "a.jpg"


def test_sorts_least_confident_first() -> None:
    item = _item(
        "b.jpg",
        (_detection("camel_line", 0.45), _detection("scout_line", 0.12)),
    )

    hits = find_confusable_detections([item], max_confidence=0.5)

    assert [round(hit.detection.confidence, 2) for hit in hits] == [0.12, 0.45]


def test_respects_custom_focus_classes() -> None:
    item = _item("c.jpg", (_detection("knight_line", 0.2), _detection("archer_line", 0.2)))

    hits = find_confusable_detections([item], focus_classes=frozenset({"archer_line"}))

    assert len(hits) == 1
    assert hits[0].detection.class_name == "archer_line"


def test_default_focus_covers_documented_cavalry_lines() -> None:
    assert "knight_line" in CAVALRY_CONFUSION_CLASSES
    assert "camel_line" in CAVALRY_CONFUSION_CLASSES
    assert "archer_line" not in CAVALRY_CONFUSION_CLASSES
