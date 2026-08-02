"""Tests for the Kalman `EntityTracker`.

The tracker measures its own elapsed wall-clock, so every test drives it
through an injected fake clock. That keeps the suite fast and, more
importantly, makes the irregular-cadence behaviour (0.3s rescans vs
multi-second LLM turns) directly expressible.
"""

from __future__ import annotations

import pytest
from core import DetectedEntity
from detection.inference.tracker import (
    _MAX_EXTRAPOLATION_S,
    _MIN_HITS_TO_TRUST,
    EntityTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOX_SIZE = 40.0
# The IoU gate caps how far an entity may move between two samples: boxes
# displaced by more than ~half their width no longer overlap enough to match.
# 15 px/s stays inside that budget even at the slowest tick used here (1.0s).
_TRACKABLE_SPEED = 15.0


class FakeClock:
    """Monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def villager_at(x: float, y: float) -> DetectedEntity:
    """A detection of the standard test size centred on (x, y)."""
    half = _BOX_SIZE / 2
    return DetectedEntity(
        id="raw",
        class_name="villager",
        bbox=(x - half, y - half, x + half, y + half),
        center=(x, y),
        confidence=0.9,
        area=_BOX_SIZE**2,
    )


def track_walking(
    tracker: EntityTracker,
    clock: FakeClock,
    *,
    speed: float,
    interval: float,
    steps: int,
) -> None:
    """Feed `steps` detections of one villager walking right at `speed` px/s."""
    for step in range(steps):
        clock.advance(interval)
        tracker.update([villager_at(100.0 + speed * interval * step, 100.0)])


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def tracker(clock: FakeClock) -> EntityTracker:
    return EntityTracker(clock=clock)


# ---------------------------------------------------------------------------
# Time base — velocity must be px/second, not px/tick
# ---------------------------------------------------------------------------


def test_velocity_is_measured_in_pixels_per_second(
    tracker: EntityTracker, clock: FakeClock
) -> None:
    track_walking(tracker, clock, speed=_TRACKABLE_SPEED, interval=0.5, steps=8)
    assert tracker.tracks[0].state[2] == pytest.approx(_TRACKABLE_SPEED, rel=0.15)


def test_velocity_survives_a_change_of_tick_rate() -> None:
    """The same physical motion sampled at two cadences reads the same speed.

    This is the regression for the fixed-dt model: advancing one step per call
    made a 0.25s tick and a 1.0s tick report velocities 4x apart for an entity
    walking at a single constant speed.
    """
    fast_clock, slow_clock = FakeClock(), FakeClock()
    fast, slow = EntityTracker(clock=fast_clock), EntityTracker(clock=slow_clock)

    track_walking(fast, fast_clock, speed=_TRACKABLE_SPEED, interval=0.25, steps=12)
    track_walking(slow, slow_clock, speed=_TRACKABLE_SPEED, interval=1.0, steps=12)

    assert fast.tracks[0].state[2] == pytest.approx(float(slow.tracks[0].state[2]), rel=0.2)


def test_long_gap_does_not_extrapolate_off_screen(tracker: EntityTracker, clock: FakeClock) -> None:
    """A multi-second stall must not fling the box away on a stale velocity."""
    track_walking(tracker, clock, speed=_TRACKABLE_SPEED, interval=0.3, steps=6)
    x_before = float(tracker.tracks[0].state[0])

    clock.advance(30.0)
    tracker.predict()

    drift = float(tracker.tracks[0].state[0]) - x_before
    assert drift <= _TRACKABLE_SPEED * _MAX_EXTRAPOLATION_S * 1.5


# ---------------------------------------------------------------------------
# Prediction confidence — must not read high on unearned identities
# ---------------------------------------------------------------------------


def test_fresh_tracks_are_not_trusted_for_prediction(
    tracker: EntityTracker, clock: FakeClock
) -> None:
    """Every new track is matched by construction — that is not confidence."""
    clock.advance(0.3)
    tracker.update([villager_at(100.0, 100.0)])
    assert tracker.prediction_confidence() == 0.0


def test_confirmed_tracks_are_trusted_for_prediction(
    tracker: EntityTracker, clock: FakeClock
) -> None:
    track_walking(tracker, clock, speed=0.0, interval=0.3, steps=_MIN_HITS_TO_TRUST)
    assert tracker.prediction_confidence() == 1.0


def test_confidence_drops_back_after_a_reset(tracker: EntityTracker, clock: FakeClock) -> None:
    track_walking(tracker, clock, speed=0.0, interval=0.3, steps=_MIN_HITS_TO_TRUST)
    tracker.reset()

    clock.advance(0.3)
    tracker.update([villager_at(500.0, 500.0)])

    assert tracker.prediction_confidence() == 0.0


def test_empty_tracker_has_no_confidence(tracker: EntityTracker) -> None:
    assert tracker.prediction_confidence() == 0.0


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_walking_entity_keeps_its_id(tracker: EntityTracker, clock: FakeClock) -> None:
    clock.advance(0.3)
    first = tracker.update([villager_at(100.0, 100.0)])[0]

    clock.advance(0.3)
    second = tracker.update([villager_at(110.0, 100.0)])[0]

    assert second.id == first.id


def test_reset_clears_every_track(tracker: EntityTracker, clock: FakeClock) -> None:
    track_walking(tracker, clock, speed=0.0, interval=0.3, steps=3)
    tracker.reset()
    assert tracker.tracks == []


def test_reset_remints_ids_rather_than_reusing_them(
    tracker: EntityTracker, clock: FakeClock
) -> None:
    """After a camera move the old ID must not reappear on a new entity."""
    clock.advance(0.3)
    before = tracker.update([villager_at(100.0, 100.0)])[0]

    tracker.reset()
    clock.advance(0.3)
    after = tracker.update([villager_at(100.0, 100.0)])[0]

    assert after.id != before.id


def test_unmatched_track_is_dropped_after_max_misses(
    tracker: EntityTracker, clock: FakeClock
) -> None:
    track_walking(tracker, clock, speed=0.0, interval=0.3, steps=3)
    for _ in range(tracker.max_misses + 1):
        clock.advance(0.3)
        tracker.update([])
    assert tracker.tracks == []
