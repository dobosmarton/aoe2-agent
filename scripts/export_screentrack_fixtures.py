"""Export golden fixtures for the screentrack Rust port from the Python tracker.

The output is deterministic, so regenerating against an unchanged reference produces
no diff. Python 3.11 is required: `sum()` of floats is compensated from 3.12 onward,
which would make the reference disagree with itself across interpreters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gameplay_agent import temporal_entities as temporal
from gameplay_agent.camera_motion import CameraMotion
from gameplay_agent.entity_snapshot import EntitySnapshot

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_REFERENCE = "apps/agent/src/temporal_entities.py"
_FULL_CANDIDATE_LIMIT = 1000
_CANDIDATE_SAMPLE = 256
_HYPOT_CASES = 4096
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64_MASK = (1 << 64) - 1
_STABLE = CameraMotion("stable")

Box = tuple[float, float, float, float]
Candidate = tuple[float, int, int]


@dataclass(frozen=True, slots=True)
class _Frame:
    entities: tuple[EntitySnapshot, ...]
    capture_id: int
    at: float
    pixels: np.ndarray
    motion: CameraMotion = _STABLE


@dataclass(slots=True)
class _Trace:
    candidates: list[Candidate] = field(default_factory=list)
    ambiguous_tracks: set[int] = field(default_factory=set)
    ambiguous_detections: set[int] = field(default_factory=set)
    matches: dict[int, int] = field(default_factory=dict)


@contextmanager
def _recording(trace: _Trace) -> Iterator[None]:
    """Capture the tracker's intermediate decisions, as the oracle test patches them."""
    candidates = temporal._candidates
    ambiguous = temporal._ambiguous_candidates
    matches = temporal._matches

    def record_candidates(
        tracks: Sequence[temporal._Track],
        detections: Sequence[temporal.PreparedDetection],
        captured_at: float,
    ) -> list[Candidate]:
        trace.candidates = candidates(tracks, detections, captured_at)
        return trace.candidates

    def record_ambiguous(items: Sequence[Candidate]) -> tuple[set[int], set[int]]:
        trace.ambiguous_tracks, trace.ambiguous_detections = ambiguous(items)
        return trace.ambiguous_tracks, trace.ambiguous_detections

    def record_matches(
        items: Sequence[Candidate], ambiguous_old: set[int], ambiguous_new: set[int]
    ) -> tuple[dict[int, int], set[int]]:
        matched, used = matches(items, ambiguous_old, ambiguous_new)
        trace.matches = matched
        return matched, used

    temporal._candidates = record_candidates
    temporal._ambiguous_candidates = record_ambiguous
    temporal._matches = record_matches
    try:
        yield
    finally:
        temporal._candidates = candidates
        temporal._ambiguous_candidates = ambiguous
        temporal._matches = matches


def _pair_digest(candidates: Sequence[Candidate]) -> str:
    """Fingerprint the candidate set for scenes too large to store in full."""
    digest = _FNV_OFFSET
    for track, detection in sorted((track, detection) for _cost, track, detection in candidates):
        digest = ((digest ^ (track << 32 | detection)) * _FNV_PRIME) & _U64_MASK
    return f"{digest:016x}"


def _track_number(identifier: str) -> int:
    return int(identifier.removeprefix("track_"))


def _motion(motion: CameraMotion) -> dict[str, object]:
    if motion.kind == "pan":
        return {"kind": "pan", "dx": float(motion.shift[0]), "dy": float(motion.shift[1])}
    return {"kind": motion.kind}


def _expected(
    trace: _Trace,
    output: tuple[EntitySnapshot, ...],
    tracker: temporal.TemporalEntities,
    *,
    accepted: bool,
) -> dict[str, object]:
    kept = trace.candidates
    if len(kept) > _FULL_CANDIDATE_LIMIT:
        kept = kept[:_CANDIDATE_SAMPLE]
    return {
        "accepted": accepted,
        "candidate_count": len(trace.candidates),
        "pair_digest": _pair_digest(trace.candidates),
        "candidates": [list(item) for item in kept],
        "ambiguous_tracks": sorted(trace.ambiguous_tracks),
        "ambiguous_detections": sorted(trace.ambiguous_detections),
        "ambiguous_count": tracker.last_ambiguous_count,
        "matches": sorted([detection, track] for detection, track in trace.matches.items()),
        "tracked": [
            {
                "track_id": _track_number(item.id),
                "label": item.class_name,
                "raw_label": item.raw_class_name,
                "agreement": item.class_agreement,
            }
            for item in output
        ],
        "track_ids": [_track_number(track.id) for track in tracker._tracks],
    }


def _replay(name: str, frames: Sequence[_Frame]) -> str:
    tracker = temporal.TemporalEntities()
    rendered: list[str] = []
    for frame in frames:
        prepared = temporal.prepare_detections(frame.entities, frame.pixels)
        trace = _Trace()
        accepted = tracker.accepts(frame.capture_id, frame.at)
        with _recording(trace):
            output = tracker.observe(
                prepared,
                capture_id=frame.capture_id,
                captured_at=frame.at,
                camera_revision=0,
                motion=frame.motion,
            )
        record = {
            "capture_id": frame.capture_id,
            "captured_at": frame.at,
            "motion": _motion(frame.motion),
            "detections": [
                {
                    "bbox": [float(value) for value in item.entity.bbox or ()],
                    "label": item.entity.class_name,
                    "confidence": item.entity.confidence,
                    "signature": list(item.appearance),
                }
                for item in prepared
            ],
            "expected": _expected(trace, output, tracker, accepted=accepted),
        }
        rendered.append(json.dumps(record, allow_nan=False, separators=(",", ":")))
    return '{"name":' + json.dumps(name) + ',"frames":[\n' + ",\n".join(rendered) + "\n]}\n"


def _noise(height: int, width: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _entity(label: str, box: Box, confidence: float = 0.8) -> EntitySnapshot:
    x0, y0, x1, y1 = box
    return EntitySnapshot("raw", label, box, ((x0 + x1) / 2, (y0 + y1) / 2), confidence)


def _stream_sequences() -> dict[str, list[_Frame]]:
    """Mirror the tracker cases in tests/test_stream_perception.py."""
    pixels = _noise(360, 640, 1)
    black = np.zeros((360, 640, 3), dtype=np.uint8)
    box: Box = (100, 100, 120, 140)
    tree = (_entity("tree", box),)
    sheep = (_entity("sheep", box),)
    villager = (_entity("villager", box),)
    return {
        "pan_preserves_identity": [
            _Frame(villager, 1, 1.0, pixels),
            _Frame(
                (_entity("villager", (108, 104, 128, 144)),),
                2,
                1.2,
                pixels,
                CameraMotion("pan", (8, 4)),
            ),
            _Frame((), 3, 1.4, pixels),
        ],
        "class_flicker": [
            _Frame(sheep, 1, 1.0, pixels),
            _Frame(sheep, 2, 1.2, pixels),
            _Frame(villager, 3, 1.3, pixels),
        ],
        "history_expires": [_Frame(tree, 1, 1.0, pixels), _Frame(tree, 2, 2.1, pixels)],
        "occlusion": [
            _Frame(villager, 1, 1.0, pixels),
            _Frame((), 2, 1.2, pixels),
            _Frame(villager, 3, 1.4, pixels),
        ],
        "ambiguous_crossing": [
            _Frame(
                (_entity("villager", box), _entity("villager", (130, 100, 150, 140))),
                1,
                1.0,
                black,
            ),
            _Frame((_entity("villager", (115, 100, 135, 140)),), 2, 1.2, black),
        ],
        "repeated_label_evidence": [
            _Frame(villager, 1, 1.0, pixels),
            _Frame(sheep, 2, 1.2, pixels),
            _Frame(sheep, 3, 1.4, pixels),
        ],
        "camera_jump_and_replay_guard": [
            _Frame(tree, 1, 1.0, pixels),
            _Frame(tree, 2, 1.2, pixels, CameraMotion("reset", reason="jump")),
            _Frame(tree, 1, 1.0, pixels),
        ],
    }


def _boundary_sequences() -> dict[str, list[_Frame]]:
    """Exact gates and windows, where a rounding slip changes a decision."""
    black = np.zeros((100, 700, 3), dtype=np.uint8)
    sequences: dict[str, list[_Frame]] = {}
    for name, old_box, inside, outside in (
        (
            "movement_boundary_small_box",
            (100.0, 50.0, 120.0, 70.0),
            (180.0, 50.0, 200.0, 70.0),
            (180.01, 50.0, 200.01, 70.0),
        ),
        (
            "movement_boundary_large_box",
            (100.0, 50.0, 300.0, 250.0),
            (315.0, 50.0, 515.0, 250.0),
            (315.01, 50.0, 515.01, 250.0),
        ),
    ):
        sequences[name] = [
            _Frame((_entity("villager", old_box),), 1, 1.0, black),
            _Frame((_entity("villager", inside), _entity("villager", outside)), 2, 2.0, black),
        ]
    box: Box = (10.0, 10.0, 30.0, 30.0)
    sequences["evidence_window_boundary"] = [
        _Frame((_entity("villager", box),), 1, 1.0, black),
        _Frame((_entity("sheep", box),), 2, 1.5, black),
        _Frame((_entity("sheep", box),), 3, 2.0, black),
        _Frame((_entity("sheep", box),), 4, 2.25, black),
    ]
    sequences["zero_confidence"] = [
        _Frame((_entity("sheep", box, 0.0),), 1, 1.0, black),
        _Frame((_entity("sheep", box, 0.0),), 2, 1.1, black),
        _Frame((_entity("villager", box, 0.0),), 3, 1.2, black),
    ]
    sequences["off_screen_boxes"] = [
        _Frame((_entity("villager", (-30.0, 10.0, -10.0, 20.0)),), 1, 1.0, _noise(100, 700, 2)),
        _Frame((_entity("villager", (-29.0, 10.0, -9.0, 20.0)),), 2, 1.2, _noise(100, 700, 2)),
    ]
    return sequences


def _exhaustive_reference() -> list[_Frame]:
    """Mirror test_matching_matches_exhaustive_reference_over_complete_sequences."""
    pixels = _noise(180, 320, 4)
    rng = np.random.default_rng(51)
    frames: list[_Frame] = []
    for capture_id in range(1, 25):
        spacing = 8 if capture_id % 4 else 35
        entities: list[EntitySnapshot] = []
        for index in range(9):
            x = float(40 + index * spacing + rng.integers(-4, 5))
            y = float(70 + rng.integers(-3, 4))
            label = "sheep" if index == 0 and capture_id % 3 == 0 else "villager"
            entities.append(_entity(label, (x, y, x + 20.0, y + 30.0)))
        motion = (
            CameraMotion("reset")
            if capture_id == 13
            else CameraMotion("pan", (2.0, 0.0))
            if capture_id % 5 == 0
            else CameraMotion("stable")
        )
        frames.append(_Frame(tuple(entities), capture_id, 1.0 + 0.2 * capture_id, pixels, motion))
    return frames


def _grid(objects: int, spacing: int, frames: int, seed: int) -> list[_Frame]:
    """The same layouts scripts/bench_temporal.py measures, at the detector's 5 Hz."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    output: list[_Frame] = []
    for frame in range(1, frames + 1):
        entities: list[EntitySnapshot] = []
        for index in range(objects):
            x = 200.0 + index % 10 * spacing + float(rng.integers(-3, 4))
            y = 200.0 + index // 10 * spacing + float(rng.integers(-3, 4))
            entities.append(_entity("villager", (x, y, x + 64.0, y + 64.0)))
        output.append(_Frame(tuple(entities), frame, 1.0 + frame / 5, pixels))
    return output


def _drifting(objects: int, seed: int) -> list[_Frame]:
    """Coloured objects under pans, resets, dropped boxes and label flicker."""
    rng = np.random.default_rng(seed)
    height, width = 360, 640
    positions = rng.uniform((20.0, 20.0), (580.0, 300.0), (objects, 2))
    velocities = rng.uniform(-30.0, 30.0, (objects, 2))
    colours = rng.integers(0, 256, (objects, 3), dtype=np.int64)
    labels = ("villager", "sheep", "militia_line")
    at = 1.0
    frames: list[_Frame] = []
    for capture_id in range(1, 41):
        delta = 0.2 + float(rng.uniform(-0.04, 0.04))
        at += delta
        kind_roll = capture_id % 17
        shift = (float(rng.integers(-12, 13)), float(rng.integers(-8, 9)))
        motion = (
            CameraMotion("reset")
            if kind_roll == 0
            else CameraMotion("pan", shift)
            if capture_id % 7 == 0
            else CameraMotion("stable")
        )
        positions += velocities * delta
        if motion.kind == "pan":
            positions += np.array(shift)
        canvas = rng.integers(0, 40, (height, width, 3), dtype=np.int64)
        entities: list[EntitySnapshot] = []
        for index in range(objects):
            x = float(positions[index, 0])
            y = float(positions[index, 1])
            box = (x, y, x + 24.5, y + 31.25)
            left, top = int(x), int(y)
            noise = rng.integers(-12, 13, (31, 24, 3), dtype=np.int64)
            patch = canvas[max(0, top) : max(0, top + 31), max(0, left) : max(0, left + 24)]
            patch[:] = np.clip(colours[index] + noise[: patch.shape[0], : patch.shape[1]], 0, 255)
            if rng.random() < 0.1:
                continue
            label = labels[index % 3] if rng.random() > 0.15 else labels[(index + 1) % 3]
            confidence = float(rng.uniform(0.3, 0.99))
            entities.append(_entity(label, box, confidence))
        frames.append(
            _Frame(tuple(entities), capture_id, at, canvas.astype(np.uint8), motion),
        )
    return frames


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _export_tracker(output: Path) -> list[str]:
    sequences: dict[str, list[_Frame]] = {
        **_stream_sequences(),
        **_boundary_sequences(),
        "exhaustive_reference": _exhaustive_reference(),
        "crowded_20": _grid(20, 6, 12, 3),
        "crowded_40": _grid(40, 6, 12, 3),
        "crowded_60": _grid(60, 6, 12, 3),
        "separated_60": _grid(60, 100, 8, 3),
        "drifting_20": _drifting(20, 11),
        "drifting_40": _drifting(40, 12),
    }
    for name, frames in sequences.items():
        _write(output / "tracker" / f"{name}.json", _replay(name, frames))
    return sorted(sequences)


def _signature_boxes(width: int, height: int, seed: int) -> list[Box]:
    rng = np.random.default_rng(seed)
    w, h = float(width), float(height)
    fixed: list[Box] = [
        (0.0, 0.0, w, h),
        (0.0, 0.0, 1.0, 1.0),
        (w - 1.0, h - 1.0, w, h),
        (10.2, 5.0, 10.8, 9.0),
        (10.9, 5.9, 12.1, 7.1),
        (-5.5, -7.25, 12.75, 9.5),
        (w - 8.5, h - 6.5, w + 20.0, h + 20.0),
        (-30.0, 10.0, -10.0, 20.0),
        (10.0, -30.0, 20.0, -10.0),
        (w + 5.0, 10.0, w + 25.0, 20.0),
        (10.0, h + 5.0, 20.0, h + 25.0),
        (-0.9, -0.9, 3.0, 3.0),
    ]
    starts = rng.uniform((0.0, 0.0), (w - 2.0, h - 2.0), (64, 2))
    sizes = rng.uniform(1.0, 80.0, (64, 2))
    drawn = [
        (float(x), float(y), float(x + bw), float(y + bh))
        for (x, y), (bw, bh) in zip(starts, sizes, strict=True)
    ]
    return fixed + drawn


def _export_signatures(output: Path) -> None:
    gradient = np.zeros((48, 64, 3), dtype=np.uint8)
    gradient[..., 0] = np.linspace(0, 255, 64, dtype=np.uint8)
    gradient[..., 1] = np.linspace(255, 0, 48, dtype=np.uint8)[:, None]
    gradient[..., 2] = 128
    images = {
        "noise_320x180": _noise(180, 320, 4),
        "gradient_64x48": gradient,
        "constant_32x24": np.full((24, 32, 3), 200, dtype=np.uint8),
    }
    cases: list[dict[str, object]] = []
    for name, pixels in images.items():
        height, width = pixels.shape[:2]
        path = output / "signatures" / f"{name}.rgb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pixels.tobytes())
        boxes = _signature_boxes(width, height, width * height)
        cases.append(
            {
                "pixels": path.name,
                "width": width,
                "height": height,
                "boxes": [list(box) for box in boxes],
                "expected": [list(temporal._appearance(pixels, box)) for box in boxes],
            }
        )
    _write(output / "signatures" / "cases.json", json.dumps(cases, allow_nan=False) + "\n")


def _export_hypot(output: Path) -> None:
    rng = random.Random(2)
    fixed = [(0.0, 0.0), (3.0, 4.0), (80.0, 0.0), (0.0, -215.0), (-5.0, 12.0), (1e-3, 1e3)]
    drawn: list[tuple[float, float]] = []
    while len(drawn) + len(fixed) < _HYPOT_CASES:
        grid = rng.random() < 0.3
        dx = rng.randrange(-2000, 2000) / 4 if grid else rng.uniform(-500.0, 500.0)
        dy = rng.randrange(-2000, 2000) / 4 if grid else rng.uniform(-500.0, 500.0)
        drawn.append((dx, dy))
    cases = [[dx, dy, math.hypot(dx, dy)] for dx, dy in fixed + drawn]
    _write(output / "hypot.json", json.dumps(cases, allow_nan=False) + "\n")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), check=True, capture_output=True, text=True
    ).stdout.strip()


class _Args(argparse.Namespace):
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="screentrack fixtures directory")
    args = parser.parse_args(namespace=_Args())
    if sys.version_info[:2] != (3, 11):
        parser.error("fixtures must come from Python 3.11, because sum() changed in 3.12")
    if _git("status", "--porcelain", "--", _REFERENCE):
        parser.error(f"commit {_REFERENCE} first, so fixtures name an exact reference")
    sequences = _export_tracker(args.output)
    _export_signatures(args.output)
    _export_hypot(args.output)
    manifest = {
        "reference_commit": _git("log", "-1", "--format=%H", "--", _REFERENCE),
        "reference_blob": _git("rev-parse", f"HEAD:{_REFERENCE}"),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": np.__version__,
        "tracker_sequences": sequences,
    }
    _write(args.output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
