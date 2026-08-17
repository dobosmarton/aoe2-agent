"""Per-phase turn latency for the game loop (ADAPTIVE-AGENT-PLAN.md 0.3).

with recorder.turn(iteration) as turn:
    with turn.phase("ocr"):
        ...
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

log = structlog.stdlib.get_logger()

MS_PER_SECOND = 1000.0
P50 = 0.50
P90 = 0.90

# Ordering for the phase percentiles. Phase names are not validated, so adding
# one costs nothing; an unlisted phase is simply absent from the snapshot.
PHASE_ORDER: tuple[str, ...] = ("capture", "ocr", "detect", "upkeep", "deliberate")


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """Turn and per-phase percentiles for one game."""

    turn_p50_ms: float = 0.0
    turn_p90_ms: float = 0.0
    turn_max_ms: float = 0.0
    phase_p50_ms: dict[str, float] = field(default_factory=dict)


def _percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of a sorted sequence; 0.0 when empty."""
    if not samples:
        return 0.0
    return samples[min(len(samples) - 1, int(len(samples) * fraction))]


@dataclass(slots=True)
class TurnTimings:
    """Phase durations for one turn, in milliseconds."""

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
        """Wall clock since the turn started, not the sum of the phases.

        The gap between phases is what unmeasured work looks like — keep it.
        """
        return (time.perf_counter() - self._started) * MS_PER_SECOND


class LatencyRecorder:
    """Collects turn and phase latency across one game."""

    def __init__(self) -> None:
        self._phase_samples: dict[str, list[float]] = defaultdict(list)
        self._turn_samples: list[float] = []

    @contextmanager
    def turn(self, iteration: int) -> Iterator[TurnTimings]:
        """Time one turn and log its breakdown, including when the turn raises."""
        timings = TurnTimings()
        try:
            yield timings
        finally:
            self._record(timings, iteration)

    def _record(self, timings: TurnTimings, iteration: int) -> None:
        total = timings.total_ms
        self._turn_samples.append(total)
        for name, elapsed in timings.phases.items():
            self._phase_samples[name].append(elapsed)
        log.info(
            "turn_latency",
            iteration=iteration,
            total_ms=round(total),
            **{f"{name}_ms": round(value) for name, value in timings.phases.items()},
        )

    def snapshot(self) -> LatencySnapshot:
        """Percentiles so far."""
        turns = sorted(self._turn_samples)
        if not turns:
            return LatencySnapshot()
        return LatencySnapshot(
            turn_p50_ms=round(_percentile(turns, P50), 1),
            turn_p90_ms=round(_percentile(turns, P90), 1),
            turn_max_ms=round(turns[-1], 1),
            phase_p50_ms={
                name: round(_percentile(sorted(self._phase_samples[name]), P50), 1)
                for name in PHASE_ORDER
                if self._phase_samples.get(name)
            },
        )
