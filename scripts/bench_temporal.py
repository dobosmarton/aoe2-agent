"""Measure temporal preparation and association until the track list stops growing.

Ambiguous tracks survive for a full history window, so a crowded scene keeps getting
slower for about one second. Warm-up and steady-state frames are reported separately:
a three-frame benchmark understates the steady-state cost several times over.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np
from gameplay_agent.camera_motion import CameraMotion
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.temporal_entities import TemporalEntities, prepare_detections

_WIDTH = 1920
_HEIGHT = 1080
_BOX = 64.0
_COLUMNS = 10
_ORIGIN = 200.0
_SEPARATED_SPACING = 100
_CROWDED_SPACING = 6
_JITTER = 3
_HISTORY_SECONDS = 1.0
_SETTLING_FRAMES = 3


@dataclass(frozen=True, slots=True)
class Timing:
    p50: float
    p95: float


@dataclass(frozen=True, slots=True)
class Result:
    layout: str
    objects: int
    rate_hz: int
    warmup_prepare_ms: Timing
    warmup_associate_ms: Timing
    steady_prepare_ms: Timing
    steady_associate_ms: Timing
    steady_total_ms: Timing


def _timing(values: list[float]) -> Timing:
    ordered = sorted(values)
    return Timing(
        round(statistics.median(ordered), 3),
        round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
    )


def _scene(rng: np.random.Generator, objects: int, spacing: int) -> tuple[EntitySnapshot, ...]:
    entities: list[EntitySnapshot] = []
    for index in range(objects):
        x = _ORIGIN + index % _COLUMNS * spacing + float(rng.integers(-_JITTER, _JITTER + 1))
        y = _ORIGIN + index // _COLUMNS * spacing + float(rng.integers(-_JITTER, _JITTER + 1))
        box = (x, y, x + _BOX, y + _BOX)
        center = (x + _BOX / 2, y + _BOX / 2)
        entities.append(EntitySnapshot(str(index), "villager", box, center, 0.8))
    return tuple(entities)


def run_case(layout: str, objects: int, rate_hz: int, frames: int, seed: int) -> Result:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    spacing = _CROWDED_SPACING if layout == "crowded" else _SEPARATED_SPACING
    settled_after = math.ceil(_HISTORY_SECONDS * rate_hz) + _SETTLING_FRAMES
    tracks = TemporalEntities()
    prepare: list[float] = []
    associate: list[float] = []
    for frame in range(1, frames + 1):
        entities = _scene(rng, objects, spacing)
        started = time.perf_counter()
        prepared = prepare_detections(entities, pixels)
        prepared_at = time.perf_counter()
        tracks.observe(
            prepared,
            capture_id=frame,
            captured_at=1.0 + frame / rate_hz,
            camera_revision=0,
            motion=CameraMotion("stable"),
        )
        finished = time.perf_counter()
        prepare.append((prepared_at - started) * 1000)
        associate.append((finished - prepared_at) * 1000)
    steady_prepare = prepare[settled_after:]
    steady_associate = associate[settled_after:]
    return Result(
        layout,
        objects,
        rate_hz,
        _timing(prepare[:settled_after]),
        _timing(associate[:settled_after]),
        _timing(steady_prepare),
        _timing(steady_associate),
        _timing([a + b for a, b in zip(steady_prepare, steady_associate, strict=True)]),
    )


def _row(result: Result) -> str:
    return (
        f"{result.layout:<10}{result.objects:>4}{result.rate_hz:>6}"
        f"{result.steady_prepare_ms.p50:>10.2f}{result.steady_prepare_ms.p95:>10.2f}"
        f"{result.steady_associate_ms.p50:>10.2f}{result.steady_associate_ms.p95:>10.2f}"
        f"{result.steady_total_ms.p50:>10.2f}{result.steady_total_ms.p95:>10.2f}"
    )


class _Args(argparse.Namespace):
    frames: int
    seed: int
    json: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args(namespace=_Args())
    results = [
        run_case(layout, objects, rate_hz, args.frames, args.seed)
        for layout in ("separated", "crowded")
        for objects in (20, 40, 60)
        for rate_hz in (5, 10)
    ]
    if args.json:
        machine = {"machine": platform.machine(), "python": platform.python_version()}
        print(json.dumps({**machine, "results": [asdict(item) for item in results]}, indent=2))
        return 0
    print("Steady-state milliseconds per frame")
    print(
        f"{'layout':<10}{'n':>4}{'Hz':>6}{'prep p50':>10}{'prep p95':>10}"
        f"{'asso p50':>10}{'asso p95':>10}{'total p50':>10}{'total p95':>10}"
    )
    for result in results:
        print(_row(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
