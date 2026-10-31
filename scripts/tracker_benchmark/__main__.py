"""Compare trackers on AoE2 frames and report whether screentrack is actually better.

    uv run --with-editable ../screentrack python scripts/tracker_benchmark --data-root .

Sequences and detections are cached under `--output`, so reruns only replay. Rules live
in `gates.toml`. The verdict is rejected if the benchmark's own controls fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import resource
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import contenders
import numpy as np
import scorecard
import sequences
from contenders import FrameRecord
from detection.inference.detector import get_detector
from gameplay_agent.camera_motion import CameraMotion
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.loops import source
from gameplay_agent.screen import NativeCapture
from gameplay_agent.sequence_replay import cache_raw_detections
from PIL import Image
from scorecard import ScoredFrame

if TYPE_CHECKING:
    from collections.abc import Callable

QUALITY_CONTENDERS = ("raw", "oracle", "iou", "python", "screentrack", "candidate")
LATENCIES = (1.0, 2.0, 5.0)
SETTLED_AFTER = 8  # About a second of frames: track lists have stopped growing.
MODEL = "packages/detection/src/inference/models/aoe2_yolo_v9.onnx"
A_A_LIMITS = (0.8, 1.25)
LAG_SLEEP_SECONDS = 0.001
LAG_ROUNDS = 40
BYTES_PER_MB = 1024 * 1024
STABILITY_WARMUP_SHARE = 0.1
CACHE = Path("logs/tracker-benchmark/v3")
RESULTS = Path("docs/tracker-benchmark-results")
SCORECARD = Path("docs/tracker-benchmark.md")


@dataclass(slots=True)
class Timing:
    """Per-frame milliseconds in matching order, so contenders stay paired."""

    prepare: list[float] = field(default_factory=list)
    associate: list[float] = field(default_factory=list)

    @property
    def total(self) -> list[float]:
        return [a + b for a, b in zip(self.prepare, self.associate, strict=True)]


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    question: str
    group: str
    blocking: bool
    passed: bool
    detail: str


def _ensure_detections(directories: list[Path], data_root: Path) -> None:
    missing = [path for path in directories if not (path / "raw.jsonl").exists()]
    if not missing:
        return
    detector = get_detector(model_path=str(data_root / MODEL), use_sahi=False, imgsz=1280)
    if detector.use_mock:
        raise RuntimeError("real detector weights are unavailable")
    for directory in missing:
        print(f"detecting {directory.name}", flush=True)
        cache_raw_detections(directory, directory / "raw.jsonl", detector)


def _band(directory: Path) -> str:
    return json.loads((directory / "scene.json").read_text())["band"]


def _weighted(
    runs: list[list[ScoredFrame]],
    metric: Callable[[list[ScoredFrame], bool | None], float],
    mobile: bool | None = None,
) -> float | None:
    """Average over scenes by objects found, so a scene without units doesn't count as 0."""
    weights = [scorecard.found_count(frames, mobile) for frames in runs]
    if not sum(weights):
        return None
    return float(np.average([metric(frames, mobile) for frames in runs], weights=weights))


def _quality(scored: dict[str, list[ScoredFrame]]) -> dict[str, float | None]:
    runs = [frames for scene, frames in scored.items() if _band_of(scene) != "targeted"]
    report: dict[str, float | None] = {
        "idf1": _weighted(runs, scorecard.idf1),
        "idf1_mobile": _weighted(runs, scorecard.idf1, True),
        "idf1_static": _weighted(runs, scorecard.idf1, False),
        "switches_per_1000": _weighted(runs, scorecard.id_switches),
        "job_error": float(np.mean([scorecard.job_error(frames) for frames in runs])),
    }
    accuracy, flips = zip(*(scorecard.label_scores(frames) for frames in runs), strict=True)
    report["label_accuracy"] = float(np.mean(accuracy))
    report["label_flips_per_1000"] = float(np.mean(flips))
    for latency in LATENCIES:
        outcomes = [scorecard.target_outcomes(frames, latency) for frames in runs]
        total = sum(item.total for item in outcomes)
        report[f"survival_{latency:g}s"] = (
            sum(item.survived for item in outcomes) / total if total else None
        )
        report[f"wrong_{latency:g}s"] = (
            sum(item.wrong for item in outcomes) / total if total else None
        )
    return report


def _band_of(scene: str) -> str:
    return scene.split("/", 1)[0]


def _paired_timing(
    scenes: list[Path], names: tuple[str, ...], repetitions: int
) -> tuple[dict[str, dict[str, Timing]], float]:
    """Time contenders on identical frames, alternating order to cancel warm-up effects."""
    timing: dict[str, dict[str, Timing]] = {name: {} for name in names}
    camera: dict[str, list[float]] = {}
    for repetition in range(repetitions):
        order = names if repetition % 2 == 0 else tuple(reversed(names))
        for name in order:
            entrant = contenders.contender(name)
            for directory in scenes:
                records, report = contenders.replay(entrant, directory)
                entry = timing[name].setdefault(_band(directory), Timing())
                settled = records[SETTLED_AFTER:]
                entry.prepare += [item.prepare_ms for item in settled]
                entry.associate += [item.associate_ms for item in settled]
                camera.setdefault(directory.name, []).append(cast("float", report["camera_ms_p95"]))
        print(f"timing repetition {repetition + 1}/{repetitions} done", flush=True)
    return timing, max(float(np.median(values)) for values in camera.values())


def _parity(records: dict[str, dict[str, list[FrameRecord]]]) -> tuple[int, str]:
    differences = 0
    example = "none"
    for scene, python in records["python"].items():
        for left, right in zip(python, records["screentrack"][scene], strict=True):
            found = scorecard.first_difference(left.entities, right.entities)
            if found is not None:
                differences += 1
                if example == "none":
                    example = f"{scene} frame {left.capture_id}: {found}"
    return differences, example


def _load_entities(directory: Path) -> tuple[tuple[EntitySnapshot, ...], NativeCapture]:
    row = json.loads((directory / "raw.jsonl").read_text().splitlines()[0])
    entities = tuple(
        EntitySnapshot(
            f"raw_{index}",
            item["class_name"],
            tuple(item["bbox"]),
            ((item["bbox"][0] + item["bbox"][2]) / 2, (item["bbox"][1] + item["bbox"][3]) / 2),
            item["confidence"],
        )
        for index, item in enumerate(row["detections"])
    )
    frame = json.loads((directory / "sequence.jsonl").read_text().splitlines()[0])
    with Image.open(directory / frame["image"]) as opened:
        image = opened.convert("RGB")
    capture = NativeCapture(image.tobytes(), image.width, image.height, 1.0, (0, 0, *image.size))
    return entities, capture


async def _loop_lag(
    name: str, entities: tuple[EntitySnapshot, ...], capture: NativeCapture
) -> float:
    """Event-loop lag while preparation runs in a worker, as `loops/source.py` runs it."""
    entrant = contenders.contender(name)
    lags: list[float] = []
    running = True

    async def heartbeat() -> None:
        while running:
            started = time.perf_counter()
            await asyncio.sleep(LAG_SLEEP_SECONDS)
            lags.append((time.perf_counter() - started - LAG_SLEEP_SECONDS) * 1000)

    original = source.prepare_detections
    source.prepare_detections = entrant.prepare
    pulse = asyncio.create_task(heartbeat())
    try:
        # Paced like the detector, so both contenders are sampled for the same wall time.
        for _ in range(LAG_ROUNDS):
            started = time.perf_counter()
            await asyncio.to_thread(source._prepare_spatial, capture, entities)
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, sequences.FRAME_SECONDS - elapsed))
    finally:
        running = False
        source.prepare_detections = original
        await pulse
    return scorecard.percentile(lags, 0.99)


def _stability(directory: Path, frames: int) -> dict[str, float]:
    """Run a long game on one scene, watching for latency drift and memory growth."""
    entities, capture = _load_entities(directory)
    entrant = contenders.contender("screentrack")
    tracker = entrant.tracker(directory)
    pixels = capture.pixels()
    motion = CameraMotion("stable")
    timings: list[float] = []
    warmup = int(frames * STABILITY_WARMUP_SHARE)
    settled_memory = 0
    for frame in range(frames):
        started = time.perf_counter()
        prepared = entrant.prepare(entities, pixels)
        tracker.observe(
            prepared,
            capture_id=frame,
            captured_at=1.0 + frame / 5,
            camera_revision=0,
            motion=motion,
        )
        timings.append((time.perf_counter() - started) * 1000)
        if frame == warmup:
            settled_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - settled_memory
    scale = 1 if sys.platform == "darwin" else 1024  # Linux reports kilobytes.
    early = scorecard.percentile(timings[warmup : warmup * 2], 0.95)
    late = scorecard.percentile(timings[-warmup:], 0.95)
    return {
        "early_p95_ms": early,
        "late_p95_ms": late,
        "memory_growth_mb": growth * scale / BYTES_PER_MB,
    }


def _gates(rules: dict[str, dict[str, object]], evidence: dict[str, object]) -> list[GateResult]:
    quality = cast("dict[str, dict[str, float]]", evidence["quality"])
    timing = cast("dict[str, dict[str, Timing]]", evidence["timing"])
    camera = cast("float", evidence["camera_p95_ms"])
    system = cast("dict[str, float]", evidence["system"])
    native = timing["screentrack"]
    results: list[GateResult] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rule = rules[name]
        results.append(
            GateResult(
                name,
                cast("str", rule["question"]),
                cast("str", rule["group"]),
                cast("bool", rule.get("blocking", True)),
                passed,
                detail,
            )
        )

    differences, example = cast("tuple[int, str]", evidence["parity"])
    add(
        "parity",
        differences <= cast("int", rules["parity"]["max_differences"]),
        f"{differences} differing frames (first: {example})",
    )
    worst = max(scorecard.percentile(entry.total, 0.95) for entry in native.values())
    limit = cast("float", rules["overhead"]["max_p95_ms"])
    python_worst = max(
        scorecard.percentile(entry.total, 0.95) for entry in timing["python"].values()
    )
    add(
        "overhead",
        camera + worst <= limit,
        f"camera {camera:.1f} + tracker {worst:.2f} = {camera + worst:.1f} ms; "
        f"Python {camera + python_worst:.1f} ms; limit {limit:g} ms",
    )
    ratios = {
        band: scorecard.bootstrap_ratio(timing["python"][band].total, entry.total)
        for band, entry in native.items()
    }
    minimum = cast("float", rules["speed"]["min_ratio"])
    add(
        "speed",
        all(low >= minimum for low, _, _ in ratios.values()),
        "; ".join(
            f"{band} {mid:.0f}x (at least {low:.0f}x)" for band, (low, mid, _) in ratios.items()
        ),
    )
    band = cast("str", rules["spec_target"]["band"])
    measured = scorecard.percentile(native[band].total, 0.95)
    target = cast("float", rules["spec_target"]["max_p95_ms"])
    add(
        "spec_target",
        measured <= target,
        f"{measured:.2f} ms in the {band} band; limit {target:g} ms",
    )

    ours = quality["screentrack"]
    survival_key = f"survival_{cast('float', rules['target_survival']['latency_seconds']):g}s"
    floor = quality[cast("str", rules["target_survival"]["baseline"])][survival_key]
    add(
        "target_survival",
        ours[survival_key] >= floor,
        f"{ours[survival_key]:.1%} against {floor:.1%} for the baseline",
    )
    wrong_key = f"wrong_{cast('float', rules['wrong_target']['latency_seconds']):g}s"
    wrong_floor = quality[cast("str", rules["wrong_target"]["baseline"])][wrong_key]
    add(
        "wrong_target",
        ours[wrong_key] <= wrong_floor,
        f"{ours[wrong_key]:.2%} against {wrong_floor:.2%} for the baseline",
    )
    recovery = cast("float | None", evidence["recovery_seconds"])
    allowed = cast("float", rules["recovery"]["max_seconds"])
    add(
        "recovery",
        recovery is not None and recovery <= allowed,
        "never recovers" if recovery is None else f"recovers in {recovery:.1f} s",
    )
    raw = quality[cast("str", rules["labels"]["baseline"])]
    add(
        "labels",
        ours["label_accuracy"] >= raw["label_accuracy"]
        and ours["label_flips_per_1000"] <= raw["label_flips_per_1000"],
        ("no change: " if ours["label_accuracy"] == raw["label_accuracy"] else "")
        + f"accuracy {ours['label_accuracy']:.2%} against {raw['label_accuracy']:.2%}; "
        f"flips {ours['label_flips_per_1000']:.1f} against {raw['label_flips_per_1000']:.1f} per 1,000",
    )
    add(
        "economy_view",
        ours["job_error"] <= raw["job_error"],
        ("no change: " if ours["job_error"] == raw["job_error"] else "")
        + f"job count error {ours['job_error']:.2f} against {raw['job_error']:.2f}",
    )
    add(
        "event_loop",
        system["loop_lag_screentrack"] <= system["loop_lag_python"],
        f"p99 lag {system['loop_lag_screentrack']:.2f} ms against {system['loop_lag_python']:.2f} ms",
    )
    drift = system["late_p95_ms"] / system["early_p95_ms"] - 1.0
    add(
        "stability",
        drift <= cast("float", rules["stability"]["max_drift"])
        and system["memory_growth_mb"] <= cast("float", rules["stability"]["max_memory_growth_mb"]),
        f"p95 drift {drift:+.1%}; memory growth {system['memory_growth_mb']:.1f} MB",
    )
    return results


def _controls(evidence: dict[str, object], limit_ms: float) -> dict[str, tuple[bool, str]]:
    """Checks on the benchmark itself. If one fails, no verdict is valid."""
    quality = cast("dict[str, dict[str, float]]", evidence["quality"])
    timing = cast("dict[str, dict[str, Timing]]", evidence["timing"])
    oracle, raw = quality["oracle"], quality["raw"]
    ratios = [
        scorecard.bootstrap_ratio(entry.total, timing["python_again"][band].total)[1]
        for band, entry in timing["python"].items()
    ]
    slowed = max(scorecard.percentile(entry.total, 0.95) for entry in timing["slowed"].values())
    ordered = all(raw["survival_2s"] <= quality[name]["survival_2s"] for name in quality)
    return {
        "oracle is perfect": (
            oracle["idf1"] == 1.0 and oracle["survival_2s"] == 1.0,
            f"IDF1 {oracle['idf1']:.3f}, survival {oracle['survival_2s']:.3f}",
        ),
        "raw scores lowest": (ordered, f"survival {raw['survival_2s']:.3f}"),
        "A/A is unbiased": (
            all(A_A_LIMITS[0] <= ratio <= A_A_LIMITS[1] for ratio in ratios),
            ", ".join(f"{ratio:.2f}x" for ratio in ratios),
        ),
        "injected slowdown is flagged": (
            slowed > limit_ms,
            f"{slowed:.1f} ms against {limit_ms:g} ms",
        ),
    }


def _versions() -> dict[str, str]:
    commit = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"), capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "commit": commit,
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "screentrack": contenders.load_screentrack().__version__,
    }


def _render(
    gates: list[GateResult],
    controls: dict[str, tuple[bool, str]],
    evidence: dict[str, object],
    scenes: list[Path],
    versions: dict[str, str],
    *,
    quick: bool,
) -> str:
    valid = all(passed for passed, _ in controls.values())
    blocking = [gate for gate in gates if gate.blocking]
    faster = all(gate.passed for gate in blocking if gate.group == "faster")
    better = all(gate.passed for gate in blocking)
    answer = "yes" if faster else "no"
    lines = ["# Tracker benchmark", ""]
    if not valid:
        lines += [
            "**Verdict: invalid. A benchmark control failed, so ignore the gates below.**",
            "",
        ]
    else:
        lines += [f"**Faster: {answer}. Better: {'yes' if better else 'no'}.**", ""]
    if quick:
        lines += ["This is a quick run, so treat it as a smoke test.", ""]
    lines += [
        f"Run on {versions['machine']} at commit `{versions['commit']}`, over {len(scenes)} "
        "generated scenes from real annotated frames, with the real detector. Units move as "
        "pasted sprites, so recorded footage is still needed for the final word.",
        "",
        "## Gates",
        "",
        "| Result | Gate | Question | Evidence |",
        "|---|---|---|---|",
    ]
    for gate in gates:
        mark = "pass" if gate.passed else ("miss" if not gate.blocking else "**FAIL**")
        lines.append(f"| {mark} | {gate.name} | {gate.question} | {gate.detail} |")
    quality = cast("dict[str, dict[str, float | None]]", evidence["quality"])
    lines += [
        "",
        "## Agent outcomes",
        "",
        "| Contender | Survival 1 s | Survival 2 s | Survival 5 s | Wrong target | IDF1 | IDF1 mobile "
        "| IDF1 static | Switches per 1,000 | Label accuracy | Flips per 1,000 | Job error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in quality.items():
        row = {key: float("nan") if value is None else value for key, value in row.items()}
        lines.append(
            f"| {name} | {row['survival_1s']:.1%} | {row['survival_2s']:.1%} | {row['survival_5s']:.1%} "
            f"| {row['wrong_2s']:.2%} | {row['idf1']:.1%} | {row['idf1_mobile']:.1%} "
            f"| {row['idf1_static']:.1%} | {row['switches_per_1000']:.0f} | {row['label_accuracy']:.2%} "
            f"| {row['label_flips_per_1000']:.1f} | {row['job_error']:.2f} |"
        )
    timing = cast("dict[str, dict[str, Timing]]", evidence["timing"])
    lines += [
        "",
        "## Milliseconds per frame at steady state",
        "",
        "| Band | Python prepare p95 | Python associate p95 | Python total p95 | screentrack prepare p95 "
        "| screentrack associate p95 | screentrack total p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for band, entry in timing["screentrack"].items():
        python = timing["python"][band]
        cells = (
            python.prepare,
            python.associate,
            python.total,
            entry.prepare,
            entry.associate,
            entry.total,
        )
        lines.append(
            f"| {band} | "
            + " | ".join(f"{scorecard.percentile(values, 0.95):.2f}" for values in cells)
            + " |"
        )
    lines += ["", "## Benchmark controls", "", "| Result | Control | Evidence |", "|---|---|---|"]
    for name, (passed, detail) in controls.items():
        lines.append(f"| {'pass' if passed else '**FAIL**'} | {name} | {detail} |")
    lines += ["", f"Camera analysis p95 was {cast('float', evidence['camera_p95_ms']):.1f} ms.", ""]
    return "\n".join(lines)


class _Args(argparse.Namespace):
    data_root: Path
    output: Path | None
    quick: bool
    repetitions: int | None
    no_report: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(), help="Checkout holding frames and weights"
    )
    parser.add_argument("--output", type=Path, help="Cache directory for scenes and detections")
    parser.add_argument("--quick", action="store_true", help="Fewer scenes and repetitions")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--no-report", action="store_true", help="Print the scorecard only")
    args = parser.parse_args(namespace=_Args())
    contenders.load_screentrack()
    rules = cast(
        "dict[str, dict[str, object]]",
        tomllib.loads((Path(__file__).parent / "gates.toml").read_text()),
    )
    output = args.output or CACHE / ("quick" if args.quick else "full")
    scenes = sequences.generate(args.data_root, output, quick=args.quick)
    _ensure_detections(scenes, args.data_root)
    targeted = sequences.write_targeted(output / "targeted")
    repetitions = args.repetitions or (2 if args.quick else 5)

    records: dict[str, dict[str, list[FrameRecord]]] = {name: {} for name in QUALITY_CONTENDERS}
    scored: dict[str, dict[str, list[ScoredFrame]]] = {name: {} for name in QUALITY_CONTENDERS}
    paths = {f"{_band(path)}/{path.name}": path for path in (*scenes, *targeted.values())}
    for name in QUALITY_CONTENDERS:
        entrant = contenders.contender(name)
        for scene, directory in paths.items():
            frames, _report = contenders.replay(entrant, directory)
            records[name][scene] = frames
            scored[name][scene] = scorecard.score_frames(frames, scorecard.read_truth(directory))
        print(f"scored {name}", flush=True)
    quality = {name: _quality(scored[name]) for name in QUALITY_CONTENDERS}

    timing, camera_p95 = _paired_timing(
        scenes, ("python", "screentrack", "python_again"), repetitions
    )
    slowed, _camera = _paired_timing(scenes, ("slowed",), 1)
    timing |= slowed

    densest = scenes[-1]
    entities, capture = _load_entities(densest)
    system = {
        "loop_lag_python": asyncio.run(_loop_lag("python", entities, capture)),
        "loop_lag_screentrack": asyncio.run(_loop_lag("screentrack", entities, capture)),
    }
    print("measuring stability", flush=True)
    frames_to_run = 1500 if args.quick else cast("int", rules["stability"]["frames"])
    system |= _stability(densest, frames_to_run)

    disturbed = 1.0 + sequences.DUPLICATE_FRAME * sequences.FRAME_SECONDS
    evidence: dict[str, object] = {
        "quality": quality,
        "timing": timing,
        "camera_p95_ms": camera_p95,
        "system": system,
        "parity": _parity(records),
        "recovery_seconds": scorecard.recovery_seconds(
            scored["screentrack"]["targeted/duplicate_box"], "gt_0", disturbed
        ),
    }
    gates = _gates(rules, evidence)
    controls = _controls(evidence, cast("float", rules["overhead"]["max_p95_ms"]))
    versions = _versions()
    text = _render(gates, controls, evidence, scenes, versions, quick=args.quick)
    print(text)
    if not args.no_report and not args.quick:
        SCORECARD.write_text(text, encoding="utf-8")
        RESULTS.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        payload = {
            "versions": versions,
            "gates": [asdict(gate) for gate in gates],
            "controls": controls,
            "quality": quality,
            "system": system,
            "camera_p95_ms": camera_p95,
            "timing_p95_ms": {
                name: {
                    band: scorecard.percentile(entry.total, 0.95) for band, entry in bands.items()
                }
                for name, bands in timing.items()
            },
        }
        path = RESULTS / f"{stamp}-{versions['machine']}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
