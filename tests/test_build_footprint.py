"""A building's whole footprint, not just its centre, must land on clear ground."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gameplay_agent.executor as ex
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_TC_BOX = (1300.0, 700.0, 1700.0, 1000.0)  # 400 px wide: a 4-tile Town Center.


@pytest.fixture(autouse=True)
def _clear_entities() -> Iterator[None]:
    yield
    ex.clear_detected_entities()


def _box(cls: str, box: tuple[float, float, float, float]) -> dict[str, object]:
    x0, y0, x1, y1 = box
    return {"class": cls, "id": cls, "bbox": box, "center": ((x0 + x1) / 2, (y0 + y1) / 2)}


def _with_town_center(*others: dict[str, object]) -> None:
    ex.set_detected_entities([_box("town_center", _TC_BOX), *others])


def test_a_farm_is_three_quarters_of_the_town_center_wide() -> None:
    _with_town_center()
    assert ex._footprint("farm") == ex._Footprint(half_width=150.0, half_height=75.0)


def test_a_clear_centre_next_to_a_mill_is_blocked_for_a_farm() -> None:
    _with_town_center(_box("mill", (2000.0, 800.0, 2150.0, 900.0)))
    assert ex._footprint_blocked((1900, 850), ex._footprint("farm"))


def test_the_same_spot_is_clear_for_a_point() -> None:
    _with_town_center(_box("mill", (2000.0, 800.0, 2150.0, 900.0)))
    assert ex._clutter_score((1900, 850)) < ex._BLOCKED_CLUTTER


def test_villagers_do_not_block_a_footprint() -> None:
    _with_town_center(_box("villager", (1880.0, 820.0, 1920.0, 880.0)))
    assert not ex._footprint_blocked((1900, 850), ex._footprint("farm"))


def test_a_blocked_footprint_ranks_below_a_clear_one() -> None:
    _with_town_center(_box("mill", (2000.0, 800.0, 2150.0, 900.0)))
    farm = ex._footprint("farm")
    first = ex._open_ground_candidates((1500, 850), footprint=farm)[0]
    assert not ex._footprint_blocked(first, farm)


def test_a_farm_placement_avoids_the_town_center() -> None:
    _with_town_center()
    point = ex.default_build_placement("a")  # Farm, in the economic menu.
    assert point is not None
    assert not ex._footprint_blocked(point, ex._footprint("farm"))
