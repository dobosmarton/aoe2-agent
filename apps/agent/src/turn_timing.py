"""Per-loop, per-phase latency for one game (ADAPTIVE-AGENT-PLAN.md 0.3).

with recorder.tick(PERCEIVE_LOOP, n) as tick:
    with tick.phase("ocr"):
        ...

One recorder holds every loop: latency is one question about one game.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

log = structlog.stdlib.get_logger()


def elapsed_ms(since: float) -> float:
    """Milliseconds since a `time.monotonic()` stamp."""
    return (time.monotonic() - since) * MS_PER_SECOND


MS_PER_SECOND = 1000.0
P50 = 0.50
P90 = 0.90
P95 = 0.95

# "turn" is the single-tick loop, until Phase 3 replaces it with the other 3.
LoopName = Literal["turn", "act", "perceive", "deliberate"]

TURN_LOOP: Final[LoopName] = "turn"
ACT_LOOP: Final[LoopName] = "act"
PERCEIVE_LOOP: Final[LoopName] = "perceive"
DELIBERATE_LOOP: Final[LoopName] = "deliberate"

# Phases each loop reports, in display order. Listing a phase is what makes it
# visible: an unlisted name is timed and logged, but dropped from the snapshot.
PHASES_BY_LOOP: Final[dict[LoopName, tuple[str, ...]]] = {
    TURN_LOOP: ("capture", "ocr", "detect", "upkeep", "deliberate"),
    ACT_LOOP: ("decide", "execute"),
    PERCEIVE_LOOP: ("capture", "ocr", "detect"),
    DELIBERATE_LOOP: ("context", "strategist", "executor"),
}


@dataclass(frozen=True, slots=True)
class LoopLatency:
    """Percentiles for one loop over one game, in milliseconds."""

    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    phase_p50_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """Every loop that recorded a tick this game."""

    loops: dict[LoopName, LoopLatency] = field(default_factory=dict)

    def of(self, loop: LoopName) -> LoopLatency:
        """That loop's percentiles, or an empty record when it never ran."""
        return self.loops.get(loop, LoopLatency())


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of a sorted sequence; 0.0 when empty.

    Nearest-rank has no interpolation, so below n=20 `int(n * P95)` clamps to
    the last index: p95 and max agree on a short game.
    """
    if not samples:
        return 0.0
    return samples[min(len(samples) - 1, int(len(samples) * fraction))]


@dataclass(slots=True)
class TickTimings:
    """Phase durations for one tick of one loop, in milliseconds."""

    phases: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one phase. Re-entering a name adds to it rather than replacing."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * MS_PER_SECOND
            self.phases[name] = self.phases.get(name, 0.0) + elapsed

    @property
    def total_ms(self) -> float:
        """Wall clock since the tick started, not the sum of the phases.

        The gap between phases is what unmeasured work looks like — keep it.
        """
        return (time.perf_counter() - self._started) * MS_PER_SECOND


class LatencyRecorder:
    """Collects tick and phase latency for every loop across one game."""

    def __init__(self) -> None:
        self._ticks: dict[LoopName, list[float]] = defaultdict(list)
        self._phases: dict[tuple[LoopName, str], list[float]] = defaultdict(list)

    @contextmanager
    def tick(self, loop: LoopName, iteration: int) -> Iterator[TickTimings]:
        """Time one tick and log its breakdown, including when the tick raises."""
        timings = TickTimings()
        try:
            yield timings
        finally:
            self._record(loop, timings, iteration)

    def _record(self, loop: LoopName, timings: TickTimings, iteration: int) -> None:
        total = timings.total_ms
        self._ticks[loop].append(total)
        for name, elapsed in timings.phases.items():
            self._phases[loop, name].append(elapsed)
        log.info(
            "loop_latency",
            loop=loop,
            iteration=iteration,
            total_ms=round(total),
            **{f"{name}_ms": round(value) for name, value in timings.phases.items()},
        )

    def snapshot(self) -> LatencySnapshot:
        """Percentiles so far, for every loop that recorded a tick."""
        return LatencySnapshot(
            loops={loop: self._loop_latency(loop) for loop in sorted(self._ticks)}
        )

    def _loop_latency(self, loop: LoopName) -> LoopLatency:
        ticks = sorted(self._ticks[loop])
        return LoopLatency(
            p50_ms=round(percentile(ticks, P50), 1),
            p90_ms=round(percentile(ticks, P90), 1),
            p95_ms=round(percentile(ticks, P95), 1),
            max_ms=round(ticks[-1], 1) if ticks else 0.0,
            phase_p50_ms={
                name: round(percentile(sorted(self._phases[loop, name]), P50), 1)
                for name in PHASES_BY_LOOP.get(loop, ())
                if self._phases.get((loop, name))
            },
        )


__all__ = [
    "ACT_LOOP",
    "DELIBERATE_LOOP",
    "MS_PER_SECOND",
    "P50",
    "P90",
    "P95",
    "PERCEIVE_LOOP",
    "PHASES_BY_LOOP",
    "TURN_LOOP",
    "LatencyRecorder",
    "LatencySnapshot",
    "LoopLatency",
    "LoopName",
    "TickTimings",
    "elapsed_ms",
    "percentile",
]
