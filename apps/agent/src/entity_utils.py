"""Utility functions for extracting attributes from detected entities.

Entities may arrive as DetectedEntity objects (from YOLO) or plain dicts
(from JSON serialization). These helpers normalize both representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from detection.inference.ownership import Owner

# The closed set of resources a villager can gather. A Literal (not a bare str) so a
# typo in a pattern table or kind lookup is a type error, not a silent no-match.
# (Implicit alias form — the repo targets Python 3.11, which lacks the PEP 695
# `type` statement.)
ResourceKind = Literal["food", "wood", "gold", "stone"]


@runtime_checkable
class _EntityLike(Protocol):
    """The object-shaped flavor of a detected entity (DetectedEntity).

    Marked `runtime_checkable` so `isinstance(entity, _EntityLike)` works
    at runtime — a DetectedEntity passes because it has all four attrs.
    The `object | dict` distinction is otherwise too loose to type-check.
    """

    id: str
    class_name: str
    center: tuple[float, float]
    confidence: float


class EntityAttrs(NamedTuple):
    """Normalized entity attributes extracted from object or dict."""

    entity_id: str
    class_name: str
    center: tuple[float, float]
    confidence: float


# Resource taxonomy shared by the reactive tier (idle-villager routing) and the
# villager-job model. `RESOURCE_KINDS` is in gather-priority order; `CLASSES_BY_KIND`
# maps each kind to the YOLO classes a villager gathers it from. Kept here (a
# dependency-light module both callers already import) so the mapping lives once.
RESOURCE_KINDS: tuple[ResourceKind, ...] = ("food", "wood", "gold", "stone")
CLASSES_BY_KIND: dict[ResourceKind, frozenset[str]] = {
    "food": frozenset({"sheep", "boar", "deer", "berry_bush", "farm"}),
    "wood": frozenset({"tree"}),
    "gold": frozenset({"gold_mine"}),
    "stone": frozenset({"stone_mine"}),
}
# Camps/drop-offs that also signal a villager's job (used by the job model, not for
# gather targeting — you can't right-click a lumber camp to chop). A mining camp
# serves both gold and stone; it maps to gold here (the common case) and stone
# villagers are tagged via stone_mine proximity instead, to keep this unambiguous.
CAMP_CLASS_BY_KIND: dict[ResourceKind, frozenset[str]] = {
    "food": frozenset({"mill"}),
    "wood": frozenset({"lumber_camp"}),
    "gold": frozenset({"mining_camp"}),
}


def _dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def nearest_class_of_kind(
    entities: list[object], kind: ResourceKind, origin: tuple[float, float] = (0.0, 0.0)
) -> str | None:
    """Concrete YOLO class of the nearest visible resource of `kind`, or None.

    Returns the *class name* (e.g. "sheep") of the closest gatherable of the given
    kind — a single string suitable for the executor's `target_class` resolution.
    "Nearest" is measured to `origin` (pass the Town Center center when known).
    """
    classes = CLASSES_BY_KIND.get(kind, frozenset())
    candidates = [a for a in (extract_attrs(e) for e in entities) if a.class_name in classes]
    if not candidates:
        return None
    return min(candidates, key=lambda a: _dist_sq(a.center, origin)).class_name


def extract_attrs(entity: object) -> EntityAttrs:
    """Extract normalized attributes from a DetectedEntity or dict."""
    if isinstance(entity, _EntityLike):
        return EntityAttrs(
            entity_id=entity.id,
            class_name=entity.class_name,
            center=entity.center,
            confidence=entity.confidence,
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
    ownership_results: dict[str, tuple[Owner, float]] | None = None,
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
