"""Utility functions for extracting attributes from detected entities.

Entities may arrive as DetectedEntity objects (from YOLO) or plain dicts
(from JSON serialization). These helpers normalize both representations.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class EntityAttrs(NamedTuple):
    """Normalized entity attributes extracted from object or dict."""

    entity_id: str
    class_name: str
    center: tuple[float, float]
    confidence: float


def extract_attrs(entity: object) -> EntityAttrs:
    """Extract normalized attributes from a DetectedEntity or dict."""
    if hasattr(entity, "id"):
        return EntityAttrs(
            entity_id=entity.id,  # type: ignore[union-attr]
            class_name=entity.class_name,  # type: ignore[union-attr]
            center=entity.center,  # type: ignore[union-attr]
            confidence=entity.confidence,  # type: ignore[union-attr]
        )
    # Dict-style access (serialized entities)
    d = entity if isinstance(entity, dict) else {}
    return EntityAttrs(
        entity_id=d.get("id", "unknown"),
        class_name=d.get("class", "unknown"),
        center=d.get("center", (0, 0)),
        confidence=d.get("confidence", 0.0),
    )


def build_entity_summary(
    entities: list[object],
    max_count: int = 20,
    ownership_results: dict[str, tuple[Any, float]] | None = None,
) -> str:
    """Build a text summary of detected entities for LLM context.

    Args:
        entities: List of DetectedEntity objects or dicts.
        max_count: Maximum number of entities to include.
        ownership_results: Optional dict mapping entity_id to (Owner, ratio) tuples.

    Returns:
        Multi-line summary string, or empty string if no entities.
    """
    if not entities:
        return ""

    lines: list[str] = []
    for entity in entities[:max_count]:
        attrs = extract_attrs(entity)
        owner_tag = ""
        if ownership_results and attrs.entity_id in ownership_results:
            owner_tag = f" [{ownership_results[attrs.entity_id][0].value}]"
        lines.append(
            f"  {attrs.entity_id}: {attrs.class_name}{owner_tag}"
            f" at ({int(attrs.center[0])},{int(attrs.center[1])})"
            f" [{attrs.confidence:.0%}]"
        )
    return "\n".join(lines)
