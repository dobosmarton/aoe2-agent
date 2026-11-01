"""Automatic identity truth, against scenes whose right answer is known by construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tracker_benchmark"))

import recorded
from recorded import Detection


def _box(x: float, y: float, size: float = 40.0) -> recorded.Box:
    return (x, y, x + size, y + size)


def _tree(x: float, y: float) -> Detection:
    return Detection("tree", _box(x, y), 0.9)


def _unit(x: float, y: float) -> Detection:
    return Detection("villager", _box(x, y), 0.8)


_TREES = ((100.0, 100.0), (400.0, 150.0), (700.0, 300.0), (250.0, 500.0))


def _still_trees(dx: float = 0.0, dy: float = 0.0) -> list[Detection]:
    return [_tree(x + dx, y + dy) for x, y in _TREES]


def test_camera_step_reads_a_pan_from_static_objects() -> None:
    assert recorded.camera_step(_still_trees(), _still_trees(-30, 12)) == pytest.approx((-30, 12))


def test_camera_step_survives_boxes_scattering_during_a_scroll() -> None:
    wobble = ((-6.0, 5.0), (5.0, -6.0), (-4.0, -3.0), (6.0, 4.0))
    after = [
        _tree(x - 60.0 + dx, y + 3.0 + dy) for (x, y), (dx, dy) in zip(_TREES, wobble, strict=True)
    ]
    step = recorded.camera_step(_still_trees(), after)
    assert step is not None
    assert step == pytest.approx((-59.0, 4.0), abs=4.0)


def test_a_far_jump_needs_more_agreement_than_a_scroll() -> None:
    few = _still_trees()
    assert recorded.camera_step(few, _still_trees(500.0, 0.0)) is None
    many = [_tree(x * 90.0, 40.0) for x in range(1, 9)]
    moved = [_tree(item.box[0] + 500.0, item.box[1]) for item in many]
    assert recorded.camera_step(many, moved) == pytest.approx((500.0, 0.0))


def test_camera_step_ignores_box_jitter() -> None:
    assert recorded.camera_step(_still_trees(), _still_trees(1.0, -1.0)) == (0.0, 0.0)


def test_camera_step_ignores_units_walking_together() -> None:
    before = [*_still_trees(), *(_unit(x, 50.0) for x in (10.0, 60.0, 110.0, 160.0, 210.0))]
    after = [*_still_trees(), *(_unit(x + 20.0, 50.0) for x in (10.0, 60.0, 110.0, 160.0, 210.0))]
    assert recorded.camera_step(before, after) == (0.0, 0.0)


def test_camera_step_reports_a_cut_when_nothing_is_shared() -> None:
    assert recorded.camera_step(_still_trees(), [_unit(5.0, 5.0)]) is None


def test_a_flickering_static_object_stays_one_track() -> None:
    frames = [
        [*_still_trees(), *([_tree(900.0, 900.0)] if index % 4 else [])] for index in range(40)
    ]
    linked = recorded.link(frames)
    assert len(linked.tracks) == len(_TREES) + 1


def test_a_walking_unit_stays_one_track() -> None:
    frames = [[*_still_trees(), _unit(50.0 + index * 4, 300.0)] for index in range(30)]
    linked = recorded.link(frames)
    assert sum(track.mobile for track in linked.tracks) == 1


def test_a_unit_survives_a_brief_detector_miss() -> None:
    frames = [
        [*_still_trees(), *([_unit(50.0 + index * 3, 300.0)] if index % 10 else [])]
        for index in range(40)
    ]
    assert sum(track.mobile for track in recorded.link(frames).tracks) == 1


def test_objects_keep_their_identity_through_a_camera_pan() -> None:
    frames = [_still_trees(-index * 25.0, 0.0) for index in range(30)]
    linked = recorded.link(frames)
    assert len(linked.tracks) == len(_TREES)
    assert linked.cuts == ()


def test_a_camera_cut_ends_tracks_instead_of_guessing() -> None:
    far = [Detection("house", _box(x, 900.0), 0.9) for x in (100.0, 300.0, 500.0, 700.0)]
    settled = recorded.MAX_LOST_FRAMES + 10
    frames = [_still_trees() for _ in range(10)] + [far for _ in range(settled + 30)]
    linked = recorded.link(frames)
    assert linked.cuts == (10 + recorded.MAX_LOST_FRAMES,)
    assert len(linked.tracks) == len(_TREES) + len(far)


def test_tracks_wait_out_a_scroll_the_camera_can_not_follow() -> None:
    scattered = [[_unit(float(x * 37), float(x * 91))] for x in range(6)]
    frames = [_still_trees() for _ in range(10)] + scattered + [_still_trees() for _ in range(10)]
    linked = recorded.link(frames)
    assert linked.cuts == ()
    assert linked.lost == frozenset(range(10, 16))
    assert sum(not track.mobile for track in linked.tracks) == len(_TREES)
    assert recorded.truth_at(linked, 12) == []


def _piece(identifier: str, frames: range, x: float) -> recorded.Track:
    track = recorded.Track(identifier, mobile=True)
    for frame in frames:
        track.boxes[frame] = _box(x + frame, 300.0)
        track.labels["villager"] += 1
    return track


def test_stitching_joins_a_unit_broken_by_a_changing_box() -> None:
    joined = recorded.stitch([_piece("a", range(10), 0.0), _piece("b", range(12, 25), 0.0)])
    assert [track.identifier for track in joined] == ["a"]
    assert sorted(joined[0].boxes)[-1] == 24


def test_stitching_never_joins_units_that_overlap_in_time() -> None:
    joined = recorded.stitch([_piece("a", range(10), 0.0), _piece("b", range(8, 20), 0.0)])
    assert len(joined) == 2


def test_stitching_prefers_the_best_overall_pairing() -> None:
    pieces = [
        _piece("left", range(10), 0.0),
        _piece("right", range(10), 60.0),
        _piece("left_again", range(12, 20), 0.0),
        _piece("right_again", range(12, 20), 60.0),
    ]
    joined = {track.identifier: track for track in recorded.stitch(pieces)}
    assert set(joined) == {"left", "right"}
    assert joined["left"].boxes[15] == _box(15.0, 300.0)
    assert joined["right"].boxes[15] == _box(75.0, 300.0)


def test_a_close_stitch_is_left_broken_rather_than_guessed() -> None:
    pieces = [
        _piece("a", range(10), 0.0),
        _piece("b", range(11, 20), 0.0),
        _piece("c", range(11, 20), 5.0),
    ]
    assert len(recorded.stitch(pieces)) == len(pieces)


def test_a_track_stops_where_two_units_could_be_confused() -> None:
    track = _piece("a", range(30), 0.0)
    track.contested = [12]
    pieces = recorded.split_at_contested(track)
    assert [sorted(piece.boxes)[-1] for piece in pieces] == [12, 29]
    assert [piece.identifier for piece in pieces] == ["a_0", "a_1"]
    assert sum(len(piece.boxes) for piece in pieces) == 30


def _farm_ghost(x: float, y: float) -> Detection:
    return Detection("villager", _box(x, y, 180.0), 0.85)


def test_a_farm_sized_villager_box_is_not_a_unit() -> None:
    assert recorded.is_unit(_unit(0.0, 0.0))
    assert not recorded.is_unit(_farm_ghost(0.0, 0.0))
    frames = [[*_still_trees(), _farm_ghost(600.0, 600.0)] for _ in range(12)]
    linked = recorded.link(frames)
    assert not any(track.mobile for track in linked.tracks)
    labels = {item["class_name"] for item in recorded.truth_at(linked, 5)}
    assert recorded.NOT_A_UNIT in labels


def test_a_real_villager_never_merges_into_a_farm_sized_box() -> None:
    frames = [[*_still_trees(), _unit(650.0, 650.0), _farm_ghost(600.0, 600.0)] for _ in range(12)]
    linked = recorded.link(frames)
    assert sum(track.mobile for track in linked.tracks) == 1
    assert sum(not track.mobile for track in linked.tracks) == len(_TREES) + 1


def test_a_one_off_false_box_never_becomes_truth() -> None:
    frames = [_still_trees() for _ in range(20)]
    frames[7] = [*frames[7], _unit(600.0, 600.0)]
    assert not any(track.mobile for track in recorded.link(frames).tracks)


def test_units_passing_each_other_are_flagged_for_review() -> None:
    frames = [
        [*_still_trees(), _unit(100.0 + index * 4, 300.0), _unit(300.0 - index * 4, 300.0)]
        for index in range(50)
    ]
    linked = recorded.link(frames)
    review = recorded._review(linked, len(frames))
    assert review["contested_frames"]
    assert sum(track.mobile for track in linked.tracks) > 2  # Split, never carried across.


def test_truth_returns_to_screen_coordinates_after_a_pan() -> None:
    frames = [_still_trees(-index * 25.0, 0.0) for index in range(12)]
    objects = recorded.truth_at(recorded.link(frames), 10)
    boxes = {tuple(item["bbox"]) for item in objects}
    assert boxes == {_box(x - 250.0, y) for x, y in _TREES if x - 250.0 + 20.0 >= 0}


def test_truth_fills_a_gap_with_the_straight_line() -> None:
    frames = [[_unit(index * 10.0, 0.0)] for index in range(20)]
    frames[10] = []
    frames_with_trees = [[*_still_trees(), *frame] for frame in frames]
    objects = recorded.truth_at(recorded.link(frames_with_trees), 10)
    walker = next(item for item in objects if item["class_name"] == "villager")
    assert walker["bbox"] == pytest.approx(list(_box(100.0, 0.0)))


def test_colour_table_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(1)
    recorded_frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    native = np.clip(recorded_frame.astype(np.int32) + 10, 0, 255).astype(np.uint8)
    table = recorded.fit_colour([(recorded_frame, native)])
    corrected = recorded.apply_colour(table, recorded_frame).astype(np.float32)
    assert np.abs(corrected - native).mean() < 6.0


def test_only_verified_units_stay_in_truth(tmp_path: Path) -> None:
    frames = [[*_still_trees(), _unit(600.0, 600.0), _unit(900.0, 600.0)] for _ in range(12)]
    linked = recorded.link(frames)
    keep = next(track.identifier for track in linked.tracks if track.mobile)
    (tmp_path / recorded.VERIFIED).write_text(json.dumps({"approved": [keep]}))
    checked = recorded._verified(linked, tmp_path)
    assert [track.identifier for track in checked.tracks if track.mobile] == [keep]
    assert sum(not track.mobile for track in checked.tracks) == len(_TREES)


def test_truth_is_unchanged_until_a_review_exists(tmp_path: Path) -> None:
    frames = [[*_still_trees(), _unit(600.0, 600.0)] for _ in range(12)]
    linked = recorded.link(frames)
    assert recorded._verified(linked, tmp_path) is linked
