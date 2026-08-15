"""Unit tests for the functional-unblock fixes (build placement, coord guard, OCR).

Cover the pure/deterministic pieces. The behavioural fixes that need a live model
(idle sweep, mill gate, tower ban) are exercised by scenario fixtures under
``scenarios/`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import gameplay_agent.executor as ex
from gameplay_agent.entity_utils import dist
from gameplay_agent.providers.strategist import _is_reliable_frame

if TYPE_CHECKING:
    import pytest

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
        placement = ex.default_build_placement("q")
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
        placement = ex.default_build_placement("q")
        # A point sitting in the eastern house cluster is more cluttered than the pick.
        assert ex._clutter_score((1900, 800)) > ex._clutter_score(placement)
    finally:
        ex.clear_detected_entities()


def test_default_build_placement_stays_in_play_area() -> None:
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        x, y = ex.default_build_placement("q")
        assert ex._in_play_area(x, y)
    finally:
        ex.clear_detected_entities()


def test_in_play_area_rejects_hud_margins() -> None:
    assert ex._in_play_area(*_TC)  # centre of the map
    assert not ex._in_play_area(8, 934)  # left screen edge
    assert not ex._in_play_area(1500, 50)  # top resource bar
    assert not ex._in_play_area(1500, 1600)  # bottom command panel


def test_window_helpers_survive_mock_rect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under headless CI, conftest shims pygetwindow with a MagicMock, so
    get_game_window_rect returns a rect whose elements are not ints. The window
    helpers must fall back to the default screen instead of raising TypeError."""
    monkeypatch.setattr(ex, "get_game_window_rect", lambda: MagicMock())
    assert ex._window_size() == ex._DEFAULT_SCREEN
    assert all(isinstance(v, int) for v in ex._play_area_bounds())
    assert ex._in_play_area(*_TC)
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        x, y = ex.default_build_placement("q")
        assert isinstance(x, int) and isinstance(y, int)
    finally:
        ex.clear_detected_entities()


def test_ocr_frame_reliability() -> None:
    assert _is_reliable_frame({"food": 200, "wood": 200, "gold": 100, "stone": 200})
    assert _is_reliable_frame({"food": 200, "wood": 200, "gold": 100})  # 3 of 4 is enough
    assert not _is_reliable_frame({"stone": 1})  # the logged garbage frame
    assert not _is_reliable_frame({"food": 200, "wood": 200})  # only 2 fields
    assert not _is_reliable_frame({})


# ---------------------------------------------------------------------------
# Drop-off camps anchor on their resource, not the town centre
# ---------------------------------------------------------------------------

_FOREST = (2400, 1100)  # well clear of the TC ring radii (280/400/520)


def _entities_with_forest() -> list[dict[str, object]]:
    """A town centre plus a tree cluster far to its south-east."""
    entities: list[dict[str, object]] = [{"class": "town_center", "center": _TC, "id": "tc"}]
    entities.extend(
        {"class": "tree", "center": (_FOREST[0] + i * 40, _FOREST[1]), "id": f"t{i}"}
        for i in range(5)
    )
    return entities


def _placement(key: str) -> tuple[int, int]:
    """The chosen point, failing the test if the build was skipped."""
    point = ex.default_build_placement(key)
    assert point is not None, f"build {key!r} was unexpectedly skipped"
    return point


def test_lumber_camp_lands_by_the_trees_not_the_town_centre() -> None:
    """The whole point: a camp at the TC carries nothing."""
    ex.set_detected_entities(_entities_with_forest())
    try:
        placement = _placement("r")
        assert dist(placement, _FOREST) < dist(placement, _TC)
    finally:
        ex.clear_detected_entities()


def test_lumber_camp_stays_within_one_ring_of_the_trees() -> None:
    """Adjacent, not merely nearer — a camp two screens away still carries nothing."""
    ex.set_detected_entities(_entities_with_forest())
    try:
        placement = _placement("r")
        assert dist(placement, _FOREST) <= max(ex.RESOURCE_RING_RADII)
    finally:
        ex.clear_detected_entities()


def test_lumber_camp_takes_the_emptiest_point_on_the_resource_ring() -> None:
    """Adjacent AND clear: the ring hugs the trees, the clutter sort finds its edge."""
    ex.set_detected_entities(_entities_with_forest())
    try:
        placement = _placement("r")
        candidates = ex._open_ground_candidates(_FOREST, ex.RESOURCE_RING_RADII)
        assert ex._clutter_score(placement) == min(ex._clutter_score(c) for c in candidates)
    finally:
        ex.clear_detected_entities()


def test_mining_camp_anchors_on_a_gold_mine() -> None:
    mine = (600, 1200)
    ex.set_detected_entities(
        [
            {"class": "town_center", "center": _TC, "id": "tc"},
            {"class": "gold_mine", "center": mine, "id": "g0"},
        ]
    )
    try:
        placement = _placement("e")
        assert dist(placement, mine) <= max(ex.RESOURCE_RING_RADII)
    finally:
        ex.clear_detected_entities()


def test_mining_camp_also_anchors_on_a_stone_mine() -> None:
    """One key, two resources — `e` serves gold and stone alike."""
    mine = (600, 1200)
    ex.set_detected_entities(
        [
            {"class": "town_center", "center": _TC, "id": "tc"},
            {"class": "stone_mine", "center": mine, "id": "s0"},
        ]
    )
    try:
        placement = _placement("e")
        assert dist(placement, mine) <= max(ex.RESOURCE_RING_RADII)
    finally:
        ex.clear_detected_entities()


def test_camp_with_no_visible_resource_returns_no_placement() -> None:
    """Skip the turn rather than spend 100 wood on a camp nobody can walk to."""
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        assert ex.default_build_placement("r") is None
    finally:
        ex.clear_detected_entities()


def test_camp_with_no_visible_resource_is_rejected_before_any_keypress() -> None:
    """Rejecting pre-flight means no build menu is opened and no ghost is left."""
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        assert ex.build_rejection("r", "test") is not None
    finally:
        ex.clear_detected_entities()


def test_mill_still_places_without_berries() -> None:
    """The mill is the farm unlock and a Feudal prerequisite; run 12 starved without one."""
    ex.set_detected_entities([{"class": "town_center", "center": _TC, "id": "tc"}])
    try:
        assert ex.default_build_placement("w") is not None
    finally:
        ex.clear_detected_entities()


def test_house_placement_is_unchanged_by_the_camp_anchoring() -> None:
    """Guards the no-regression claim for every key outside the anchor table."""
    ex.set_detected_entities(_entities_with_forest())
    try:
        assert ex.default_build_placement("q") == ex.default_build_placement("")
    finally:
        ex.clear_detected_entities()
