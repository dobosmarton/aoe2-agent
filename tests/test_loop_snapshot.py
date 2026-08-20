"""Unit tests for loops/snapshot.py — the frame and the slot the 3 clocks share."""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError

import pytest
from gameplay_agent.loops.snapshot import Perception, Slot

# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def test_a_fresh_frame_is_almost_zero_milliseconds_old() -> None:
    assert Perception().age_ms < 100.0


def test_age_grows_with_the_clock() -> None:
    """The act loop skips a stale rule on this number, so it must move."""
    frame = Perception()
    time.sleep(0.01)
    assert frame.age_ms >= 10.0


def test_a_frame_is_frozen() -> None:
    """Three loops share one frame by reference; a writable field would tear."""
    frame = Perception()
    with pytest.raises(FrozenInstanceError):
        frame.alarm = True  # type: ignore[misc]


def test_a_frame_starts_with_no_hud_reading() -> None:
    """ResourceReadings is total=False, so "not read yet" is the empty dict."""
    assert Perception().hud_readings == {}


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------


def test_an_empty_slot_reads_none() -> None:
    """The act loop starts before the first frame exists."""
    assert Slot[int]().get() is None


def test_put_replaces_the_value() -> None:
    slot = Slot[int](1)
    slot.put(2)
    assert slot.get() == 2


def test_get_leaves_the_value_in_place() -> None:
    """The act loop reads the same frame every tick until a newer one lands."""
    slot = Slot[int](1)
    slot.get()
    assert slot.get() == 1
