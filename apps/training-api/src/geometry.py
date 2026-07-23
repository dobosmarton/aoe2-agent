"""Annotation geometry value objects and their serialization boundary.

Canonical unit is **absolute pixels**, top-left origin — the same convention as
the detector's `DetectionResult.bbox` and COCO. YOLO's normalized format is an
external representation; every conversion to/from it lives in this module so the
boundary condition sits in exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeAlias, assert_never, cast

GeomType: TypeAlias = Literal["bbox", "polygon"]


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in absolute pixels, top-left origin."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class Polygon:
    """Closed polygon as ordered (x, y) vertices in absolute pixels."""

    points: tuple[tuple[float, float], ...]


Geometry: TypeAlias = BBox | Polygon


def geom_type_of(geom: Geometry) -> GeomType:
    match geom:
        case BBox():
            return "bbox"
        case Polygon():
            return "polygon"
        case _ as unreachable:
            assert_never(unreachable)


def to_coords_json(geom: Geometry) -> str:
    """Serialize just the coordinates; the geom type is stored alongside."""
    match geom:
        case BBox(x, y, w, h):
            return json.dumps([x, y, w, h])
        case Polygon(points):
            return json.dumps([[px, py] for px, py in points])
        case _ as unreachable:
            assert_never(unreachable)


def from_coords_json(geom_type: GeomType, raw: str) -> Geometry:
    """Inverse of `to_coords_json`, discriminated by the stored `geom_type`."""
    return geometry_from_coords(geom_type, cast("object", json.loads(raw)))


def geometry_from_coords(geom_type: GeomType, coords: object) -> Geometry:
    """Build a `Geometry` from already-parsed coordinates (the client-request path).

    Shares the exact same validation as `from_coords_json`, so a bbox posted as
    JSON and a bbox read back from the DB go through one boundary check.
    """
    match geom_type:
        case "bbox":
            return _bbox_from_parsed(coords)
        case "polygon":
            return _polygon_from_parsed(coords)
        case _ as unreachable:
            assert_never(unreachable)


def _bbox_from_parsed(parsed: object) -> BBox:
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"bbox coords must be a 4-element list, got {parsed!r}")
    x, y, w, h = (_as_float(v) for v in parsed)
    return BBox(x=x, y=y, w=w, h=h)


def _polygon_from_parsed(parsed: object) -> Polygon:
    if not isinstance(parsed, list):
        raise ValueError(f"polygon coords must be a list of points, got {parsed!r}")
    points: list[tuple[float, float]] = []
    for vertex in parsed:
        if not isinstance(vertex, list) or len(vertex) != 2:
            raise ValueError(f"polygon vertex must be [x, y], got {vertex!r}")
        points.append((_as_float(vertex[0]), _as_float(vertex[1])))
    return Polygon(points=tuple(points))


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def bbox_from_yolo(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> BBox:
    """Convert a normalized YOLO box (center form, 0..1) to an absolute-pixel BBox."""
    abs_w = w * img_w
    abs_h = h * img_h
    return BBox(
        x=cx * img_w - abs_w / 2,
        y=cy * img_h - abs_h / 2,
        w=abs_w,
        h=abs_h,
    )


def bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert an absolute-pixel BBox to normalized YOLO (cx, cy, w, h)."""
    return (
        (bbox.x + bbox.w / 2) / img_w,
        (bbox.y + bbox.h / 2) / img_h,
        bbox.w / img_w,
        bbox.h / img_h,
    )
