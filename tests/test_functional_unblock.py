"""Unit tests for the functional-unblock fixes (build placement, coord guard, OCR).

Cover the pure/deterministic pieces. The behavioural fixes that need a live model
(idle sweep, mill gate, tower ban) are exercised by scenario fixtures under
``scenarios/`` instead.
"""

from __future__ import annotations

import gameplay_agent.executor as ex
from gameplay_agent.providers.strategist import _is_reliable_frame

_TC = (1500, 800)


def _entities_clustered_east() -> list[dict[str, object]]:
    """A town centre with houses packed to its east — the west stays open."""
    entities: list[dict[str, object]] = [{"class": "town_center", "center": _TC, "id": "tc"}]
    entities.extend(
        {"class": "house", "center": (1880 + i * 12, 800), "id": f"h{i}"} for i in range(6)
    )
    return entities


def test_default_build_placement_avoids_tc_tile() -> None:
    ex.set_detected_entities(_entities_clustered_east())
    try:
        placement = ex.default_build_placement()
        assert placement != _TC  # never the TC tile itself (always blocked)
        # It returns the emptiest candidate on the ring around the TC.
        candidates = ex._open_ground_candidates(_TC)
        assert ex._clutter_score(placement) == min(ex._clutter_score(c) for c in candidates)
        assert ex._clutter_score(placement) == 0  # open ground exists away from the cluster
    finally:
        ex.clear_detected_entities()


def test_open_ground_prefers_emptier_direction() -> None:
    ex.set_detected_entities(_entities_clustered_east())
    try:
        placement = ex.default_build_placement()
        # A point sitting in the eastern house cluster is more cluttered than the pick.
        assert ex._clutter_score((1900, 800)) > ex._clutter_score(placement)
    finally:
        ex.clear_detected_entities()


def test_default_build_placement_stays_in_play_area() -> None:
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        x, y = ex.default_build_placement()
        assert ex._in_play_area(x, y)
    finally:
        ex.clear_detected_entities()


def test_in_play_area_rejects_hud_margins() -> None:
    assert ex._in_play_area(*_TC)  # centre of the map
    assert not ex._in_play_area(8, 934)  # left screen edge
    assert not ex._in_play_area(1500, 50)  # top resource bar
    assert not ex._in_play_area(1500, 1600)  # bottom command panel


def test_ocr_frame_reliability() -> None:
    assert _is_reliable_frame({"food": 200, "wood": 200, "gold": 100, "stone": 200})
    assert _is_reliable_frame({"food": 200, "wood": 200, "gold": 100})  # 3 of 4 is enough
    assert not _is_reliable_frame({"stone": 1})  # the logged garbage frame
    assert not _is_reliable_frame({"food": 200, "wood": 200})  # only 2 fields
    assert not _is_reliable_frame({})
