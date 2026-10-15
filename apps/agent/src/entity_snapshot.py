"""Immutable detected-entity values at the gameplay-agent boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

MAX_SPATIAL_AGE_SECONDS = 1.0


@runtime_checkable
class _EntityLike(Protocol):
    """Object-shaped detector output consumed by the normalizer."""

    id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class EntitySnapshot:
    """Stable entity shape shared by perception, summaries, and overlays."""

    id: str
    class_name: str
    bbox: tuple[float, float, float, float] | None
    center: tuple[float, float]
    confidence: float
    raw_class_name: str | None = None
    class_agreement: float | None = None
    capture_id: int = -1
    camera_revision: int = 0
    captured_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the executor cache's serialized representation."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "center": self.center,
            "confidence": self.confidence,
            "raw_class_name": self.raw_class_name or self.class_name,
            "class_agreement": self.class_agreement,
            "capture_id": self.capture_id,
            "camera_revision": self.camera_revision,
            "captured_at": self.captured_at,
        }


def snapshot_entity(entity: object) -> EntitySnapshot:
    """Normalize one detector object or serialized cache entry."""
    if isinstance(entity, EntitySnapshot):
        return entity
    if isinstance(entity, _EntityLike):
        return EntitySnapshot(
            id=_text(entity.id, "unknown"),
            class_name=_text(entity.class_name, "unknown"),
            bbox=_box(entity.bbox),
            center=_point(entity.center),
            confidence=_number(entity.confidence),
        )
    if isinstance(entity, Mapping):
        capture_id = entity.get("capture_id")
        camera_revision = entity.get("camera_revision")
        agreement = entity.get("class_agreement")
        raw_class = entity.get("raw_class_name")
        captured_at = entity.get("captured_at")
        return EntitySnapshot(
            id=_text(entity.get("id"), "unknown"),
            class_name=_text(entity.get("class_name", entity.get("class")), "unknown"),
            bbox=_box(entity.get("bbox")),
            center=_point(entity.get("center")),
            confidence=_number(entity.get("confidence")),
            raw_class_name=_text(raw_class, "unknown") if raw_class is not None else None,
            class_agreement=(_number(agreement) if agreement is not None else None),
            capture_id=capture_id if isinstance(capture_id, int) else -1,
            camera_revision=camera_revision if isinstance(camera_revision, int) else 0,
            captured_at=_number(captured_at) if captured_at is not None else None,
        )
    return EntitySnapshot(
        id="unknown",
        class_name="unknown",
        bbox=None,
        center=(0.0, 0.0),
        confidence=0.0,
    )


def snapshot_entities(entities: Sequence[object]) -> tuple[EntitySnapshot, ...]:
    """Normalize a complete detector result at one explicit boundary."""
    return tuple(snapshot_entity(entity) for entity in entities)


def _text(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if isfinite(number) else 0.0


def _point(value: object) -> tuple[float, float]:
    values = _numbers(value, expected=2)
    return (values[0], values[1]) if values is not None else (0.0, 0.0)


def _box(value: object) -> tuple[float, float, float, float] | None:
    values = _numbers(value, expected=4)
    if values is None or values[0] >= values[2] or values[1] >= values[3]:
        return None
    return (values[0], values[1], values[2], values[3])


def _numbers(value: object, *, expected: int) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != expected:
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not isfinite(number):
            return None
        numbers.append(number)
    return tuple(numbers)


__all__ = ["MAX_SPATIAL_AGE_SECONDS", "EntitySnapshot", "snapshot_entities", "snapshot_entity"]
