"""Turn a Mac screen recording of AoE2 into a sequence the benchmark can score.

Recording from the host keeps the VM untouched and gives 15 fps, while the tracker
only sees every third frame. Identity truth comes from all of them, because units
barely move between frames. Camera motion comes from static objects, so the truth
does not depend on the camera code or trackers it judges.

    python scripts/tracker_benchmark/recorded.py colour --pair clip.mov frame.png --out c.npy
    python scripts/tracker_benchmark/recorded.py prepare --video m.mov --colour c.npy \
        --start 360 --seconds 60 --output logs/tracker-benchmark/recorded/still
    python scripts/tracker_benchmark/recorded.py score logs/tracker-benchmark/recorded/still
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import contenders
import httpx
import numpy as np
import scorecard
from core import DetectedEntity
from detection.inference.postprocess import nms
from gameplay_agent.camera_motion import MOBILE_CLASSES
from gameplay_agent.sequence_replay import _overlap
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

Box = tuple[float, float, float, float]
Step = tuple[float, float]

CAPTURE_WIDTH = 3024
CAPTURE_HEIGHT = 1672
WINDOW_TOP = 142  # Where the game starts inside the recorded Mac display.
JPEG_QUALITY = 85  # What the detector receives live.
DETECTOR_SIZE = 1280
NMS_OVERLAP = 0.5  # RemoteDetector's client-side suppression.
COLOUR_BINS = 33
SAMPLE_STEP = 3  # 15 fps recording, 5 Hz tracker.
LINK_OVERLAP = 0.3
RELABEL_OVERLAP = 0.6  # A static box may change label only when clearly the same object.
MOBILE_GAP_FRAMES = 15  # One second. Static objects can't walk off, so they have no limit.
MIN_TRACK_FRAMES = 8
MIN_CAMERA_VOTES = 3
CAMERA_BIN_PX = 4
CAMERA_SNAP_PX = 2.0  # Box jitter, not a scroll: stops rounding adding up over a clip.
VELOCITY_FRAMES = 5
CONTESTED_MARGIN = 0.1
SEED = 0
MAX_FIT_PIXELS = 600_000
REPORT_NAMES = ("raw", "oracle", "iou", "python", "screentrack", "candidate")


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    box: Box
    score: float


@dataclass(slots=True)
class Track:
    identifier: str
    mobile: bool
    boxes: dict[int, Box] = field(default_factory=dict)
    labels: Counter[str] = field(default_factory=Counter)
    contested: list[int] = field(default_factory=list)

    @property
    def last(self) -> int:
        return max(self.boxes)

    def predicted(self, frame: int) -> Box:
        """Where the object should be, in world coordinates."""
        frames = sorted(self.boxes)[-VELOCITY_FRAMES:]
        latest = self.boxes[frames[-1]]
        if not self.mobile or len(frames) < 2:
            return latest
        first = self.boxes[frames[0]]
        span = frames[-1] - frames[0]
        ahead = frame - frames[-1]
        dx = (latest[0] - first[0]) / span * ahead
        dy = (latest[1] - first[1]) / span * ahead
        return (latest[0] + dx, latest[1] + dy, latest[2] + dx, latest[3] + dy)


@dataclass(frozen=True, slots=True)
class Linked:
    tracks: tuple[Track, ...]
    shifts: tuple[Step, ...]  # Screen position of the world origin, per frame.
    cuts: tuple[int, ...]


def colour_key(rgb: np.ndarray) -> np.ndarray:
    cells = (rgb.astype(np.int32) * (COLOUR_BINS - 1) + 127) // 255
    return cells[..., 0] * COLOUR_BINS**2 + cells[..., 1] * COLOUR_BINS + cells[..., 2]


def _polynomial(values: np.ndarray) -> np.ndarray:
    return np.c_[values, values**2, np.ones(len(values))]


def fit_colour(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Learn recorded-to-VM colours from paused frames seen by both.

    Cells never seen fall back to a smooth fit, so unusual colours stay sensible.
    """
    cells = COLOUR_BINS**3
    totals = np.zeros((cells, 3))
    counts = np.zeros(cells)
    for recorded, native in pairs:
        keys = colour_key(recorded).ravel()
        for channel in range(3):
            totals[:, channel] += np.bincount(
                keys, weights=native[..., channel].ravel(), minlength=cells
            )
        counts += np.bincount(keys, minlength=cells)
    sources = np.concatenate([recorded.reshape(-1, 3) for recorded, _ in pairs]) / 255.0
    targets = np.concatenate([native.reshape(-1, 3) for _, native in pairs]) / 255.0
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(sources), min(MAX_FIT_PIXELS, len(sources)), replace=False)
    weights = np.linalg.lstsq(_polynomial(sources[picks]), targets[picks], rcond=None)[0]
    axis = np.linspace(0.0, 1.0, COLOUR_BINS)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    table = _polynomial(lattice) @ weights * 255.0
    seen = counts > 0
    table[seen] = (totals[seen].T / counts[seen]).T
    return np.clip(table, 0.0, 255.0).astype(np.float32)


def apply_colour(table: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return np.clip(table[colour_key(rgb)].round(), 0, 255).astype(np.uint8)


def _centre(box: Box) -> Step:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _shifted(box: Box, step: Step, sign: float = 1.0) -> Box:
    dx, dy = step[0] * sign, step[1] * sign
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def camera_step(previous: Sequence[Detection], current: Sequence[Detection]) -> Step | None:
    """How static objects moved on screen, or None when too few are shared.

    Every same-label pair votes for the offset between them. Wrong pairs scatter, while
    objects that really are the same agree, so the largest group is the camera.
    """
    offsets: list[Step] = []
    for before in previous:
        if before.label in MOBILE_CLASSES:
            continue
        origin = _centre(before.box)
        for after in current:
            if after.label == before.label:
                target = _centre(after.box)
                offsets.append((target[0] - origin[0], target[1] - origin[1]))
    if not offsets:
        return None
    votes = Counter((round(dx / CAMERA_BIN_PX), round(dy / CAMERA_BIN_PX)) for dx, dy in offsets)
    (bin_x, bin_y), count = votes.most_common(1)[0]
    if count < MIN_CAMERA_VOTES:
        return None
    agreeing = [
        offset
        for offset in offsets
        if abs(offset[0] - bin_x * CAMERA_BIN_PX) <= CAMERA_BIN_PX
        and abs(offset[1] - bin_y * CAMERA_BIN_PX) <= CAMERA_BIN_PX
    ]
    dx, dy = np.median(np.asarray(agreeing), axis=0).tolist()
    if abs(dx) <= CAMERA_SNAP_PX and abs(dy) <= CAMERA_SNAP_PX:
        return 0.0, 0.0
    return dx, dy


def _can_link(track: Track, detection: Detection, overlap: float) -> bool:
    if track.mobile != (detection.label in MOBILE_CLASSES) or overlap < LINK_OVERLAP:
        return False
    return track.mobile or detection.label in track.labels or overlap >= RELABEL_OVERLAP


def _assign(
    open_tracks: Sequence[Track], detections: Sequence[Detection], world: Sequence[Box], frame: int
) -> dict[int, Track]:
    """Greedy, best overlap first, noting when a choice was a close call."""
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(open_tracks):
        predicted = track.predicted(frame)
        for index, box in enumerate(world):
            overlap = _overlap(predicted, box)
            if _can_link(track, detections[index], overlap):
                pairs.append((overlap, track_index, index))
    pairs.sort(reverse=True)
    taken: dict[int, Track] = {}
    used: set[int] = set()
    for overlap, track_index, index in pairs:
        if track_index in used or index in taken:
            continue
        track = open_tracks[track_index]
        rivals = (
            other
            for other in pairs
            if (other[1] == track_index) != (other[2] == index)
            and other[0] >= overlap - CONTESTED_MARGIN
        )
        if track.mobile and next(rivals, None) is not None:
            track.contested.append(frame)
        taken[index] = track
        used.add(track_index)
    return taken


def link(detections: Sequence[Sequence[Detection]]) -> Linked:
    """Follow every object through the full-rate frames, in world coordinates."""
    tracks: list[Track] = []
    open_tracks: list[Track] = []
    shifts: list[Step] = []
    cuts: list[int] = []
    shift: Step = (0.0, 0.0)
    for frame, current in enumerate(detections):
        if frame:
            step = camera_step(detections[frame - 1], current)
            if step is None:
                cuts.append(frame)
                open_tracks = []
            else:
                shift = (shift[0] + step[0], shift[1] + step[1])
        shifts.append(shift)
        world = [_shifted(item.box, shift, -1.0) for item in current]
        open_tracks = [
            track
            for track in open_tracks
            if not track.mobile or frame - track.last <= MOBILE_GAP_FRAMES
        ]
        taken = _assign(open_tracks, current, world, frame)
        for index, detection in enumerate(current):
            track = taken.get(index)
            if track is None:
                track = Track(f"track_{len(tracks)}", detection.label in MOBILE_CLASSES)
                tracks.append(track)
                open_tracks.append(track)
            track.boxes[frame] = world[index]
            track.labels[detection.label] += 1
    kept = tuple(track for track in tracks if len(track.boxes) >= MIN_TRACK_FRAMES)
    return Linked(kept, tuple(shifts), tuple(cuts))


def _world_box(track: Track, frame: int) -> Box | None:
    """The recorded box, or the straight line between its neighbours across a gap."""
    if frame in track.boxes:
        return track.boxes[frame]
    frames = sorted(track.boxes)
    if not frames[0] < frame < frames[-1]:
        return None
    before = max(item for item in frames if item < frame)
    after = min(item for item in frames if item > frame)
    share = (frame - before) / (after - before)
    return tuple(
        low + (high - low) * share
        for low, high in zip(track.boxes[before], track.boxes[after], strict=True)
    )  # type: ignore[return-value]


def truth_at(linked: Linked, frame: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for track in linked.tracks:
        box = _world_box(track, frame)
        if box is None:
            continue
        x0, y0, x1, y1 = _shifted(box, linked.shifts[frame])
        centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
        if 0 <= centre_x < CAPTURE_WIDTH and 0 <= centre_y < CAPTURE_HEIGHT:
            objects.append(
                {
                    "track_id": track.identifier,
                    "class_name": track.labels.most_common(1)[0][0],
                    "bbox": [x0, y0, x1, y1],
                }
            )
    return objects


def _review(linked: Linked, count: int) -> dict[str, object]:
    """Where automatic identity could be wrong, so people only look there."""
    contested = sorted({frame for track in linked.tracks for frame in track.contested})
    crossings: list[dict[str, object]] = []
    mobile = [track for track in linked.tracks if track.mobile]
    for frame in range(count):
        here = [(track, _world_box(track, frame)) for track in mobile]
        boxes = [(track, box) for track, box in here if box is not None]
        crossings.extend(
            {"frame": frame, "tracks": [left.identifier, right.identifier]}
            for index, (left, box) in enumerate(boxes)
            for right, other in boxes[index + 1 :]
            if _overlap(box, other) >= LINK_OVERLAP
        )
    pairs = {tuple(item["tracks"]) for item in crossings}  # type: ignore[arg-type]
    return {
        "frames": count,
        "tracks": len(linked.tracks),
        "mobile_tracks": len(mobile),
        "camera_cuts": list(linked.cuts),
        "contested_frames": contested,
        "overlapping_pairs": len(pairs),
        "overlap_events": crossings,
    }


def video_frames(video: Path, start: float, seconds: float, top: int) -> Iterator[np.ndarray]:
    pipeline = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(seconds), "-i", str(video),
        "-fps_mode", "passthrough",
        "-vf", f"crop={CAPTURE_WIDTH}:{CAPTURE_HEIGHT}:0:{top},format=rgb24",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    size = CAPTURE_WIDTH * CAPTURE_HEIGHT * 3
    with subprocess.Popen(pipeline, stdout=subprocess.PIPE, bufsize=size) as process:
        assert process.stdout is not None
        while len(raw := process.stdout.read(size)) == size:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)


def frame_times(video: Path, start: float, seconds: float) -> list[float]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"{start}%+{seconds}",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(line.split(",")[0]) for line in probe.stdout.splitlines() if line.strip()]


def _jpeg(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _detect(client: httpx.Client, server: str, jpeg: bytes) -> list[Detection]:
    """Ask the same server the live agent uses, then suppress as its client does."""
    response = client.post(
        f"{server}/detect",
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        params={"imgsz": DETECTOR_SIZE},
    )
    response.raise_for_status()
    entities = [
        DetectedEntity(
            id=str(index),
            class_name=item["class_name"],
            bbox=tuple(item["bbox"]),
            center=tuple(item["center"]),
            confidence=item["confidence"],
        )
        for index, item in enumerate(response.json()["detections"])
    ]
    kept = nms(entities, iou_threshold=NMS_OVERLAP)
    return [Detection(item.class_name, tuple(item.bbox), item.confidence) for item in kept]


def _dense_row(frame: int, at: float, detections: Sequence[Detection]) -> str:
    objects = [
        {"class_name": item.label, "bbox": list(item.box), "confidence": item.score}
        for item in detections
    ]
    return json.dumps({"frame": frame, "at": at, "detections": objects})


def _detect_clip(arguments: argparse.Namespace, directory: Path) -> None:
    table = np.load(arguments.colour)
    times = frame_times(arguments.video, arguments.start, arguments.seconds)
    frames = video_frames(arguments.video, arguments.start, arguments.seconds, arguments.top)
    lines: list[str] = []
    with httpx.Client(timeout=30) as client:
        for index, (at, recorded) in enumerate(zip(times, frames, strict=False)):
            jpeg = _jpeg(apply_colour(table, recorded))
            if index % SAMPLE_STEP == 0:
                (directory / f"frame_{index // SAMPLE_STEP:05d}.jpg").write_bytes(jpeg)
            lines.append(_dense_row(index, at, _detect(client, arguments.server, jpeg)))
            if index % 150 == 0:
                print(f"{directory.name}: frame {index}", flush=True)
    (directory / "dense.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_dense(directory: Path) -> tuple[list[float], list[list[Detection]]]:
    times: list[float] = []
    frames: list[list[Detection]] = []
    for line in (directory / "dense.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        times.append(row["at"])
        frames.append(
            [
                Detection(item["class_name"], tuple(item["bbox"]), item["confidence"])
                for item in row["detections"]
            ]
        )
    return times, frames


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_sequence(directory: Path) -> dict[str, object]:
    """Build the replay's files from cached detections, so truth rules can change freely."""
    times, frames = _read_dense(directory)
    linked = link(frames)
    picked = range(0, len(frames), SAMPLE_STEP)
    start = times[0]
    _write_jsonl(
        directory / "sequence.jsonl",
        [
            {
                "capture_id": index,
                "captured_at_monotonic": 1.0 + times[frame] - start,
                "image": f"frame_{index:05d}.jpg",
                "input_events": [],
            }
            for index, frame in enumerate(picked)
        ],
    )
    _write_jsonl(
        directory / "raw.jsonl",
        [
            {
                "capture_id": index,
                "detections": [
                    {"class_name": item.label, "bbox": list(item.box), "confidence": item.score}
                    for item in frames[frame]
                ],
            }
            for index, frame in enumerate(picked)
        ],
    )
    _write_jsonl(
        directory / "labels.jsonl",
        [
            {"capture_id": index, "objects": truth_at(linked, frame)}
            for index, frame in enumerate(picked)
        ],
    )
    review = _review(linked, len(frames))
    (directory / "review.json").write_text(json.dumps(review), encoding="utf-8")
    return review


def _percent(value: float) -> str:
    return f"{value * 100:5.1f}%"


def score(directory: Path, names: Sequence[str]) -> dict[str, dict[str, float]]:
    truth = scorecard.read_truth(directory)
    results: dict[str, dict[str, float]] = {}
    records: dict[str, list[contenders.FrameRecord]] = {}
    for name in names:
        records[name], _ = contenders.replay(contenders.contender(name), directory)
        frames = scorecard.score_frames(records[name], truth)
        results[name] = {
            "survival_1s": scorecard.target_outcomes(frames, 1.0).survival,
            "survival_2s": scorecard.target_outcomes(frames, 2.0).survival,
            "survival_5s": scorecard.target_outcomes(frames, 5.0).survival,
            "survival_2s_mobile": scorecard.target_outcomes(frames, 2.0, mobile=True).survival,
            "survival_2s_static": scorecard.target_outcomes(frames, 2.0, mobile=False).survival,
            "wrong_2s": scorecard.target_outcomes(frames, 2.0).wrong_rate,
            "idf1": scorecard.idf1(frames),
            "switches": scorecard.id_switches(frames),
        }
    if {"python", "screentrack"} <= set(records):
        differing = sum(
            scorecard.first_difference(left.entities, right.entities) is not None
            for left, right in zip(records["python"], records["screentrack"], strict=True)
        )
        results["parity"] = {"differing_frames": float(differing)}
    return results


def _print_scores(results: dict[str, dict[str, float]]) -> None:
    print("contender      1 s     2 s     5 s  mobile  static  wrong   IDF1  switches/1000")
    for name, row in results.items():
        if name == "parity":
            continue
        print(
            f"{name:12s} {_percent(row['survival_1s'])} {_percent(row['survival_2s'])} "
            f"{_percent(row['survival_5s'])} {_percent(row['survival_2s_mobile'])} "
            f"{_percent(row['survival_2s_static'])} {_percent(row['wrong_2s'])} "
            f"{_percent(row['idf1'])} {row['switches']:8.0f}"
        )
    if "parity" in results:
        print(
            f"python and screentrack differ on {results['parity']['differing_frames']:.0f} frames"
        )


def _paused_frame(video: Path, top: int) -> np.ndarray:
    """The middle of a short recording made while the game was paused."""
    middle = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout) / 2  # fmt: skip
    return next(video_frames(video, middle, 1.0, top))


def _colour_command(arguments: argparse.Namespace) -> None:
    pairs = [
        (
            _paused_frame(Path(video), arguments.top),
            np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8),
        )
        for video, frame in arguments.pair
    ]
    np.save(arguments.out, fit_colour(pairs))


def _prepare_command(arguments: argparse.Namespace) -> None:
    directory: Path = arguments.output
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "dense.jsonl").exists():
        _detect_clip(arguments, directory)
    review = write_sequence(directory)
    summary = {key: value for key, value in review.items() if key != "overlap_events"}
    summary["contested_frames"] = len(review["contested_frames"])  # type: ignore[arg-type]
    print(json.dumps(summary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    colour = commands.add_parser("colour")
    colour.add_argument("--pair", nargs=2, action="append", required=True)
    colour.add_argument("--out", type=Path, required=True)
    colour.add_argument("--top", type=int, default=WINDOW_TOP)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--colour", type=Path, required=True)
    prepare.add_argument("--start", type=float, required=True)
    prepare.add_argument("--seconds", type=float, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--top", type=int, default=WINDOW_TOP)
    prepare.add_argument("--server", default="http://127.0.0.1:8420")
    rate = commands.add_parser("score")
    rate.add_argument("directory", type=Path)
    rate.add_argument("--names", nargs="+", default=list(REPORT_NAMES))
    arguments = parser.parse_args()
    if arguments.command == "colour":
        _colour_command(arguments)
    elif arguments.command == "prepare":
        _prepare_command(arguments)
    else:
        results = score(arguments.directory, arguments.names)
        _print_scores(results)
        (arguments.directory / "scores.json").write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
