"""Unit tests for gameplay_agent/entity_utils.py."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gameplay_agent.entity_utils import EntityAttrs, build_entity_summary, extract_attrs

# ---------------------------------------------------------------------------
# Test infra: a minimal stand-in for DetectedEntity (which is a CV-only class
# we don't import here to avoid pulling YOLO/Torch into the test path).
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntity:
    id: str
    class_name: str
    center: tuple[float, float]
    confidence: float


class _Owner(Enum):
    """Minimal stand-in for detection.inference.ownership.Owner."""

    SELF = "self"
    ENEMY = "enemy"


# ---------------------------------------------------------------------------
# extract_attrs
# ---------------------------------------------------------------------------


def test_extract_attrs_from_object_with_attrs() -> None:
    e = _FakeEntity(id="sheep_1", class_name="sheep", center=(100.0, 200.0), confidence=0.9)
    attrs = extract_attrs(e)
    assert attrs == EntityAttrs(
        entity_id="sheep_1", class_name="sheep", center=(100.0, 200.0), confidence=0.9
    )


def test_extract_attrs_from_dict() -> None:
    e = {"id": "tree_3", "class": "tree", "center": (50, 60), "confidence": 0.8}
    attrs = extract_attrs(e)
    assert attrs.entity_id == "tree_3"
    assert attrs.class_name == "tree"
    assert attrs.center == (50, 60)
    assert attrs.confidence == 0.8


def test_extract_attrs_dict_uses_class_key_not_class_name() -> None:
    """Dicts use 'class' (matching JSON serialization), objects use class_name."""
    d = {"id": "x", "class": "villager", "center": (0, 0), "confidence": 1.0}
    assert extract_attrs(d).class_name == "villager"


def test_extract_attrs_dict_with_missing_fields_uses_defaults() -> None:
    attrs = extract_attrs({"id": "x"})
    assert attrs.class_name == "unknown"
    assert attrs.center == (0, 0)
    assert attrs.confidence == 0.0


def test_extract_attrs_non_dict_non_object_falls_back_to_unknown() -> None:
    """A None/string/number gets the all-unknown default."""
    attrs = extract_attrs(None)  # type: ignore[arg-type]
    assert attrs.entity_id == "unknown"
    assert attrs.class_name == "unknown"


# ---------------------------------------------------------------------------
# build_entity_summary
# ---------------------------------------------------------------------------


def test_build_entity_summary_empty_returns_empty_string() -> None:
    assert build_entity_summary([]) == ""


def test_build_entity_summary_single_entity_format() -> None:
    e = _FakeEntity(id="s1", class_name="sheep", center=(100.0, 200.0), confidence=0.85)
    summary = build_entity_summary([e])
    # Format: "  <id>: <class> at (<x>,<y>) [<confidence>%]"
    assert "s1: sheep at (100,200)" in summary
    assert "[85%]" in summary


def test_build_entity_summary_multiple_lines() -> None:
    entities = [
        _FakeEntity(id="s1", class_name="sheep", center=(10, 20), confidence=0.9),
        _FakeEntity(id="t1", class_name="tree", center=(30, 40), confidence=0.7),
    ]
    summary = build_entity_summary(entities)
    lines = summary.split("\n")
    assert len(lines) == 2
    assert "s1: sheep" in lines[0]
    assert "t1: tree" in lines[1]


def test_build_entity_summary_respects_max_count() -> None:
    entities = [
        _FakeEntity(id=f"e{i}", class_name="tree", center=(i, i), confidence=0.5) for i in range(50)
    ]
    summary = build_entity_summary(entities, max_count=5)
    assert summary.count("\n") == 4  # 5 lines = 4 newlines
    assert "e0:" in summary
    assert "e4:" in summary
    assert "e5:" not in summary  # truncated


def test_build_entity_summary_default_max_count_20() -> None:
    entities = [
        _FakeEntity(id=f"e{i}", class_name="tree", center=(0, 0), confidence=0.5) for i in range(50)
    ]
    summary = build_entity_summary(entities)
    assert summary.count("\n") == 19  # 20 entries


def test_build_entity_summary_with_ownership_tag() -> None:
    """Entities found in ownership_results get a [<owner.value>] suffix."""
    entities = [
        _FakeEntity(id="enemy_villager", class_name="villager", center=(0, 0), confidence=0.9),
    ]
    ownership = {"enemy_villager": (_Owner.ENEMY, 0.85)}
    summary = build_entity_summary(entities, ownership_results=ownership)
    assert "[enemy]" in summary


def test_build_entity_summary_no_ownership_tag_when_id_absent() -> None:
    """Entities not in ownership_results show no owner tag."""
    entities = [_FakeEntity(id="x", class_name="sheep", center=(0, 0), confidence=0.9)]
    ownership = {"OTHER_ENTITY": (_Owner.ENEMY, 0.5)}
    summary = build_entity_summary(entities, ownership_results=ownership)
    assert "[" not in summary.replace("[90%]", "")  # only the confidence bracket remains


def test_build_entity_summary_works_on_dicts() -> None:
    entities = [
        {"id": "s1", "class": "sheep", "center": (100, 200), "confidence": 0.9},
    ]
    summary = build_entity_summary(entities)
    assert "s1: sheep at (100,200)" in summary


def test_build_entity_summary_floats_truncate_to_int() -> None:
    """Coords are formatted as int — fractional precision is dropped."""
    e = _FakeEntity(id="x", class_name="sheep", center=(123.7, 456.4), confidence=0.5)
    summary = build_entity_summary([e])
    assert "(123,456)" in summary
