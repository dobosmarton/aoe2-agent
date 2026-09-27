"""Detected entities cross into the agent through one typed shape."""

from core import DetectedEntity
from gameplay_agent.entity_snapshot import EntitySnapshot, snapshot_entity


def test_detector_object_becomes_an_immutable_snapshot() -> None:
    entity = DetectedEntity(
        id="sheep_0",
        class_name="sheep",
        bbox=(1.0, 2.0, 3.0, 4.0),
        center=(2.0, 3.0),
        confidence=0.9,
    )

    assert snapshot_entity(entity) == EntitySnapshot(
        id="sheep_0",
        class_name="sheep",
        bbox=(1.0, 2.0, 3.0, 4.0),
        center=(2.0, 3.0),
        confidence=0.9,
    )


def test_serialized_cache_entry_preserves_overlay_fields() -> None:
    snapshot = snapshot_entity(
        {
            "id": "tree_2",
            "class": "tree",
            "bbox": [10, 20, 30, 40],
            "center": [20, 30],
            "confidence": 0.75,
        }
    )

    assert (snapshot.bbox, snapshot.center, snapshot.class_name) == (
        (10.0, 20.0, 30.0, 40.0),
        (20.0, 30.0),
        "tree",
    )


def test_malformed_values_degrade_to_non_renderable_unknowns() -> None:
    snapshot = snapshot_entity({"id": 7, "class": None, "bbox": [1, 2, "bad", 4], "center": "bad"})

    assert (snapshot.id, snapshot.class_name, snapshot.bbox, snapshot.center) == (
        "unknown",
        "unknown",
        None,
        (0.0, 0.0),
    )
