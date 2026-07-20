"""Property + unit tests for training_api.geometry.

The geometry module is the single conversion boundary (pixels ↔ YOLO, objects ↔
JSON), so it gets the most rigorous coverage: round-trips must be lossless.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from training_api.geometry import (
    BBox,
    Polygon,
    bbox_from_yolo,
    bbox_to_yolo,
    from_coords_json,
    geom_type_of,
    to_coords_json,
)

_coord = st.floats(min_value=-10_000, max_value=10_000, allow_nan=False, allow_infinity=False)
_norm = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


@given(x=_coord, y=_coord, w=_coord, h=_coord)
def test_bbox_json_roundtrip(x: float, y: float, w: float, h: float) -> None:
    box = BBox(x=x, y=y, w=w, h=h)
    assert from_coords_json("bbox", to_coords_json(box)) == box


@given(points=st.lists(st.tuples(_coord, _coord), min_size=3, max_size=12))
def test_polygon_json_roundtrip(points: list[tuple[float, float]]) -> None:
    poly = Polygon(points=tuple(points))
    assert from_coords_json("polygon", to_coords_json(poly)) == poly


@given(cx=_norm, cy=_norm, w=_norm, h=_norm)
def test_yolo_pixel_roundtrip(cx: float, cy: float, w: float, h: float) -> None:
    img_w, img_h = 1280, 720
    box = bbox_from_yolo(cx, cy, w, h, img_w, img_h)
    back = bbox_to_yolo(box, img_w, img_h)
    assert back == pytest.approx((cx, cy, w, h))


def test_bbox_from_yolo_centers_the_box() -> None:
    box = bbox_from_yolo(0.5, 0.5, 0.2, 0.4, 1000, 500)
    assert box == BBox(x=400.0, y=150.0, w=200.0, h=200.0)


def test_geom_type_of_discriminates() -> None:
    assert geom_type_of(BBox(0, 0, 1, 1)) == "bbox"
    assert geom_type_of(Polygon(points=((0, 0), (1, 0), (1, 1)))) == "polygon"


def test_from_coords_json_rejects_malformed_bbox() -> None:
    with pytest.raises(ValueError, match="4-element"):
        from_coords_json("bbox", "[1, 2, 3]")
