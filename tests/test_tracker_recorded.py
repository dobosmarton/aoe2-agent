"""Automatic identity truth, against scenes whose right answer is known by construction."""

from __future__ import annotations

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
    frames = [_still_trees() for _ in range(10)] + [far for _ in range(10)]
    linked = recorded.link(frames)
    assert linked.cuts == (10,)
    assert len(linked.tracks) == len(_TREES) + len(far)


def test_a_one_off_false_box_never_becomes_truth() -> None:
    frames = [_still_trees() for _ in range(20)]
    frames[7] = [*frames[7], _unit(600.0, 600.0)]
    assert not any(track.mobile for track in recorded.link(frames).tracks)


def test_units_passing_each_other_are_flagged_for_review() -> None:
    frames = [
        [*_still_trees(), _unit(100.0 + index * 4, 300.0), _unit(300.0 - index * 4, 300.0)]
        for index in range(50)
    ]
    review = recorded._review(recorded.link(frames), len(frames))
    assert review["overlap_events"]
    assert review["contested_frames"]


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
