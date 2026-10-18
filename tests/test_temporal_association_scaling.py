"""Optimized association must make the same conservative choices as exhaustive matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gameplay_agent import temporal_entities as temporal
from gameplay_agent.camera_motion import CameraMotion
from gameplay_agent.entity_snapshot import EntitySnapshot

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pytest


def _entity(index: int, x: float, y: float, label: str = "villager") -> EntitySnapshot:
    box = (x, y, x + 20.0, y + 30.0)
    return EntitySnapshot(str(index), label, box, (x + 10.0, y + 15.0), 0.8)


def _exhaustive_candidates(
    tracks: Sequence[temporal._Track],
    detections: Sequence[temporal.PreparedDetection],
    captured_at: float,
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for old_index, track in enumerate(tracks):
        for new_index, detection in enumerate(detections):
            box = detection.entity.bbox
            assert box is not None
            cost = temporal._matching_cost(
                track, box, detection.appearance, captured_at - track.last_seen_at
            )
            if cost is not None:
                candidates.append((cost, old_index, new_index))
    return sorted(candidates)


def _exhaustive_ambiguity(
    candidates: Sequence[tuple[float, int, int]],
) -> tuple[set[int], set[int]]:
    ambiguous_new = {
        new_index
        for cost, _old, new_index in candidates
        if sum(
            other_new == new_index and abs(other_cost - cost) < 0.15
            for other_cost, _other_old, other_new in candidates
        )
        > 1
    }
    ambiguous_old = {
        old_index
        for cost, old_index, _new in candidates
        if sum(
            other_old == old_index and abs(other_cost - cost) < 0.15
            for other_cost, other_old, _other_new in candidates
        )
        > 1
    }
    return ambiguous_old, ambiguous_new


def test_matching_matches_exhaustive_reference_over_complete_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.random.default_rng(4).integers(0, 256, (180, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(51)
    optimized = temporal.TemporalEntities()
    reference = temporal.TemporalEntities()
    for capture_id in range(1, 25):
        # Alternating crowded and spread-out scenes exercise unmatched history.
        spacing = 8 if capture_id % 4 else 35
        boxes = tuple(
            _entity(
                index,
                float(40 + index * spacing + rng.integers(-4, 5)),
                float(70 + rng.integers(-3, 4)),
                "sheep" if index == 0 and capture_id % 3 == 0 else "villager",
            )
            for index in range(9)
        )
        prepared = temporal.prepare_detections(boxes, pixels)
        motion = (
            CameraMotion("reset")
            if capture_id == 13
            else CameraMotion("pan", (2.0, 0.0))
            if capture_id % 5 == 0
            else CameraMotion("stable")
        )
        kwargs = {
            "capture_id": capture_id,
            "captured_at": 1.0 + 0.2 * capture_id,
            "camera_revision": capture_id // 5,
            "motion": motion,
        }
        result = optimized.observe(prepared, **kwargs)
        with monkeypatch.context() as patched:
            patched.setattr(temporal, "_candidates", _exhaustive_candidates)
            patched.setattr(temporal, "_ambiguous_candidates", _exhaustive_ambiguity)
            expected = reference.observe(prepared, **kwargs)
        assert result == expected
        assert optimized.last_ambiguous_count == reference.last_ambiguous_count


def test_lower_ranked_close_costs_remain_ambiguous() -> None:
    candidates = [(0.0, 0, 0), (0.10, 1, 1), (0.20, 0, 2), (0.24, 2, 1), (0.25, 0, 3)]
    assert temporal._ambiguous_candidates(candidates) == _exhaustive_ambiguity(candidates)
    assert temporal._ambiguous_candidates(candidates) == ({0}, {1})


def test_equal_costs_are_ambiguous_but_the_exact_threshold_is_not() -> None:
    equal = [(0.0, 0, 0), (0.0, 0, 1), (0.0, 1, 0)]
    assert temporal._ambiguous_candidates(equal) == ({0}, {0})
    assert temporal._ambiguous_candidates(equal) == _exhaustive_ambiguity(equal)

    threshold = [(0.0, 0, 0), (0.15, 0, 1), (0.15, 1, 0)]
    assert temporal._ambiguous_candidates(threshold) == (set(), set())
    assert temporal._ambiguous_candidates(threshold) == _exhaustive_ambiguity(threshold)


def test_spatial_pruning_keeps_exact_movement_boundary_and_large_boxes() -> None:
    def prepared(index: int, box: tuple[float, float, float, float]) -> temporal.PreparedDetection:
        x0, y0, x1, y1 = box
        entity = EntitySnapshot(str(index), "villager", box, ((x0 + x1) / 2, (y0 + y1) / 2), 0.8)
        return temporal.PreparedDetection(entity, (0.0,) * 6)

    for old_box, boundary_box, outside_box in (
        # At one second, a 20-pixel box permits exactly 15 + 65 = 80 pixels.
        ((100.0, 50.0, 120.0, 70.0), (180.0, 50.0, 200.0, 70.0), (180.01, 50.0, 200.01, 70.0)),
        # A 200-pixel box permits 150 + 65 = 215 pixels.
        ((100.0, 50.0, 300.0, 250.0), (315.0, 50.0, 515.0, 250.0), (315.01, 50.0, 515.01, 250.0)),
    ):
        track = temporal._Track("t", old_box, (0.0,) * 6, (), 1.0)
        detections = (prepared(0, boundary_box), prepared(1, outside_box))
        expected = _exhaustive_candidates((track,), detections, 2.0)
        assert [new_index for _cost, _old_index, new_index in expected] == [0]
        assert temporal._candidates((track,), detections, 2.0) == expected


def test_candidate_work_is_spatially_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    track = temporal._Track("t", (100.0, 50.0, 120.0, 70.0), (0.0,) * 6, (), 1.0)
    detections = tuple(
        temporal.PreparedDetection(_entity(index, float(100 + index * 200), 50.0), (0.0,) * 6)
        for index in range(1000)
    )
    original = temporal._matching_cost
    calls = 0

    def counted(
        track: temporal._Track,
        box: tuple[float, float, float, float],
        appearance: tuple[float, ...],
        elapsed: float,
    ) -> float | None:
        nonlocal calls
        calls += 1
        return original(track, box, appearance, elapsed)

    monkeypatch.setattr(temporal, "_matching_cost", counted)
    temporal._candidates((track,), detections, 1.2)
    assert calls == 1


def test_ambiguity_detection_walks_candidates_once() -> None:
    class CountedCandidates(list[tuple[float, int, int]]):
        def __init__(self) -> None:
            super().__init__((float(index), 0, index) for index in range(1000))
            self.iterations = 0

        def __iter__(self) -> Iterator[tuple[float, int, int]]:
            self.iterations += 1
            return super().__iter__()

    candidates = CountedCandidates()
    temporal._ambiguous_candidates(candidates)
    assert candidates.iterations == 1
