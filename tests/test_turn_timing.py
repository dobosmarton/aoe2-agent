"""Unit tests for turn_timing.py — the per-loop latency record.

Phase 3 judges the act loop on p95, so the percentile itself needs a test.
"""

from __future__ import annotations

import time

import pytest
from gameplay_agent.turn_timing import (
    ACT_LOOP,
    P50,
    P90,
    P95,
    TURN_LOOP,
    LatencyRecorder,
    LatencySnapshot,
    percentile,
)

# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_an_empty_sample_reads_zero() -> None:
    assert percentile([], P50) == 0.0


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(P50, 6.0), (P90, 10.0), (P95, 10.0)],
    ids=["p50", "p90", "p95"],
)
def test_nearest_rank_over_ten_samples(fraction: float, expected: float) -> None:
    assert percentile([float(n) for n in range(1, 11)], fraction) == expected


def test_a_single_sample_is_everypercentile() -> None:
    assert percentile([7.0], P95) == 7.0


# ---------------------------------------------------------------------------
# LatencyRecorder
# ---------------------------------------------------------------------------


def test_a_loop_that_never_ran_reads_empty() -> None:
    """The act columns must not invent a number before Phase 3 lands."""
    assert LatencySnapshot().of(ACT_LOOP).p95_ms == 0.0


def test_a_recorded_tick_reports_a_duration() -> None:
    assert _recorder(TURN_LOOP).snapshot().of(TURN_LOOP).p50_ms > 0.0


def test_each_loop_is_recorded_apart() -> None:
    """One recorder holds every loop; act latency must not include perceive."""
    recorder = LatencyRecorder()
    with recorder.tick(ACT_LOOP, 0):
        pass
    assert recorder.snapshot().of(TURN_LOOP).p50_ms == 0.0


def test_only_the_loops_phases_reach_the_snapshot() -> None:
    """`ocr` belongs to perceive, not act, so it is dropped from an act record."""
    recorder = LatencyRecorder()
    with recorder.tick(ACT_LOOP, 0) as tick, tick.phase("ocr"):
        time.sleep(0.002)
    assert recorder.snapshot().of(ACT_LOOP).phase_p50_ms == {}


def test_a_tick_is_recorded_even_when_it_raises() -> None:
    """A crashed tick is the one whose latency matters most. It is instant, so
    the duration rounds to 0.0 — the loop appearing at all is the assertion."""
    recorder = LatencyRecorder()
    with pytest.raises(RuntimeError), recorder.tick(TURN_LOOP, 0):
        raise RuntimeError("tick failed")
    assert TURN_LOOP in recorder.snapshot().loops


def test_re_entering_a_phase_adds_to_it() -> None:
    """The loop times `upkeep` on 2 branches; the second must not replace the first."""
    recorder = LatencyRecorder()
    with recorder.tick(TURN_LOOP, 0) as tick:
        for _ in range(2):
            with tick.phase("ocr"):
                time.sleep(0.002)
    assert tick.phases["ocr"] >= 4.0


def test_the_log_names_the_loop(capsys: pytest.CaptureFixture[str]) -> None:
    """Three loops share one event name, so the label is what separates them."""
    _recorder(ACT_LOOP)
    assert "loop=act" in capsys.readouterr().out


def _recorder(loop: str, ticks: int = 3) -> LatencyRecorder:
    """A recorder holding `ticks` measurable ticks of `loop`."""
    recorder = LatencyRecorder()
    for iteration in range(ticks):
        with recorder.tick(loop, iteration):
            time.sleep(0.002)  # an empty tick rounds to 0.0 ms
    return recorder
