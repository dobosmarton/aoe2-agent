"""Clicks on things that don't move may use older evidence, while the camera stays put."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def ledger() -> Iterator[ex.ActionLedger]:
    ledger = ex.ActionLedger()
    token = ex.bind_ledger(ledger)
    ledger.spatial_valid = True
    try:
        yield ledger
    finally:
        ex.unbind_ledger(token)
        ex.clear_detected_entities()


def _seen(cls: str, age: float, *, camera: int = 0) -> None:
    ex.set_detected_entities(
        [
            {
                "id": "target",
                "class": cls,
                "bbox": (90, 90, 110, 110),
                "center": (100, 100),
                "confidence": 0.9,
                "capture_id": 7,
                "captured_at": time.monotonic() - age,
                "camera_revision": camera,
            }
        ]
    )


def _resolve(cls: str) -> str:
    detail, _coords = ex._resolve_coords({"target_id": "target", "expected_class": cls})
    return detail


@pytest.mark.parametrize("cls", ["tree", "gold_mine", "stone_mine", "house", "sheep", "cow"])
def test_a_resource_or_building_may_be_clicked_from_1_5_s_old_evidence(
    ledger: ex.ActionLedger, cls: str
) -> None:
    _seen(cls, 1.5)
    assert _resolve(cls) == ""


@pytest.mark.parametrize("cls", ["villager", "deer", "boar"])
def test_a_moving_target_keeps_the_one_second_limit(ledger: ex.ActionLedger, cls: str) -> None:
    _seen(cls, 1.5)
    assert "too old" in _resolve(cls)


def test_static_evidence_still_expires(ledger: ex.ActionLedger) -> None:
    _seen("tree", 2.5)
    assert "too old" in _resolve("tree")


def test_static_evidence_from_another_camera_position_is_refused(
    ledger: ex.ActionLedger,
) -> None:
    ledger.camera_revision = 3
    _seen("tree", 1.5, camera=2)
    assert "too old" in _resolve("tree")


def test_fresh_evidence_is_used_for_any_target(ledger: ex.ActionLedger) -> None:
    _seen("sheep", 0.3)
    assert _resolve("sheep") == ""


def test_a_building_placement_counts_as_static() -> None:
    assert ex._static_click({"auto_placement": True, "building_key": "q"})
    assert not ex._static_click({"type": "click", "x": 3, "y": 4})


def _villager(identifier: str, center: tuple[int, int], confidence: float) -> dict[str, object]:
    x, y = center
    return {
        "id": identifier,
        "class": "villager",
        "bbox": (x - 10, y - 15, x + 10, y + 15),
        "center": center,
        "confidence": confidence,
    }


_TC = {"id": "tc", "class": "town_center", "bbox": (400, 300, 800, 600), "center": (600, 450)}


def test_a_villager_on_the_town_center_is_not_clicked(ledger: ex.ActionLedger) -> None:
    # The stronger detection stands on the Town Center; clicking it selects the TC.
    ex.set_detected_entities(
        [_TC, _villager("on_tc", (600, 500), 0.95), _villager("in_field", (1000, 700), 0.8)]
    )
    target = ex._verified_villager_target()
    assert target is not None and target[0] == "in_field"


def test_a_villager_on_a_building_is_used_when_it_is_the_only_one(
    ledger: ex.ActionLedger,
) -> None:
    ex.set_detected_entities([_TC, _villager("on_tc", (600, 500), 0.95)])
    target = ex._verified_villager_target()
    assert target is not None and target[0] == "on_tc"


def test_a_villager_over_an_animal_is_not_clicked(ledger: ex.ActionLedger) -> None:
    # Game 2: a builder click on a shepherd selected the cow carcass at its feet.
    cow = {"id": "cow", "class": "cow", "bbox": (985, 690, 1030, 720), "center": (1007, 705)}
    ex.set_detected_entities(
        [cow, _villager("shepherd", (1000, 700), 0.95), _villager("lumberjack", (300, 700), 0.8)]
    )
    target = ex._verified_villager_target()
    assert target is not None and target[0] == "lumberjack"


def test_a_villager_click_lands_on_its_body(ledger: ex.ActionLedger) -> None:
    ex.set_detected_entities([_villager("worker", (1000, 700), 0.9)])
    detail, coords = ex._resolve_coords(
        {"target_id": "worker", "expected_class": "villager", "expected_coords": (1000, 700)}
    )
    assert detail == ""
    # 35% of the way down the 30-pixel box (685-715), above its centre.
    assert coords == (1000, 695)
