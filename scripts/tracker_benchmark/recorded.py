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
import random
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import contenders
import httpx
import numpy as np
import scorecard
from core import DetectedEntity
from detection.inference.postprocess import nms
from gameplay_agent.camera_motion import MOBILE_CLASSES
from gameplay_agent.sequence_replay import _overlap
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

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
MAX_UNIT_SIZE_PX = 80.0  # The detector also labels whole farms `villager`, at high confidence.
NOT_A_UNIT = "not_a_unit"
MOBILE_GAP_FRAMES = 6  # 0.4 s. Static objects can't walk off, so they have no limit.
MIN_TRACK_FRAMES = 8
STRIP_MIN_FRAMES = 15  # Sampled frames. Shorter tracks can't reach a 2 s target anyway.
STRIP_COLUMNS = 9
STRIP_ROWS = 4
STRIP_CELL_PX = 240
STRIP_CROP_PX = 120
MIN_CAMERA_VOTES = 4
CAMERA_TOLERANCE_PX = 12.0  # Boxes scatter this much while the view scrolls.
MAX_SCROLL_PX = 150.0  # Further than this in one frame is a jump, not a scroll.
MIN_JUMP_VOTES = 6
JUMP_VOTE_SHARE = 0.4  # Chance agreement is easy over a whole screen of offsets.
CAMERA_SNAP_PX = 2.0  # Box jitter, not a scroll: stops rounding adding up over a clip.
MAX_LOST_FRAMES = 20  # How long to wait for a fast scroll to settle before calling it a cut.
STITCH_MAX_GAP_FRAMES = 6
STITCH_BASE_PX = 20.0
STITCH_PX_PER_FRAME = 3.0
STITCH_MARGIN_PX = 10.0
UNLINKABLE = 1e9
VELOCITY_FRAMES = 5
CONTESTED_MARGIN = 0.1
SEED = 0
GATES = Path(__file__).parent / "gates.toml"
VERIFIED = "verified.json"
MAX_FIT_PIXELS = 600_000
REPORT_NAMES = ("raw", "oracle", "iou", "python", "screentrack")


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
    lost: frozenset[int] = frozenset()  # Frames whose camera position is unknown.


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


def is_unit(detection: Detection) -> bool:
    """Whether this box can walk. A unit-labelled box the size of a farm can't."""
    x0, y0, x1, y1 = detection.box
    return detection.label in MOBILE_CLASSES and max(x1 - x0, y1 - y0) <= MAX_UNIT_SIZE_PX


def _centre(box: Box) -> Step:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _shifted(box: Box, step: Step, sign: float = 1.0) -> Box:
    dx, dy = step[0] * sign, step[1] * sign
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _static_offsets(previous: Sequence[Detection], current: Sequence[Detection]) -> list[Step]:
    offsets: list[Step] = []
    for before in previous:
        if before.label in MOBILE_CLASSES:
            continue
        origin = _centre(before.box)
        for after in current:
            if after.label == before.label:
                target = _centre(after.box)
                offsets.append((target[0] - origin[0], target[1] - origin[1]))
    return offsets


def camera_step(previous: Sequence[Detection], current: Sequence[Detection]) -> Step | None:
    """How static objects moved on screen, or None when they don't agree.

    Every same-label pair proposes an offset. Wrong pairs scatter, while objects that
    really are the same agree, so the largest group is the camera.
    """
    offsets = np.asarray(_static_offsets(previous, current))
    if not offsets.size:
        return None
    near = np.abs(offsets[:, None] - offsets[None, :]).max(axis=2) <= CAMERA_TOLERANCE_PX
    votes = near.sum(axis=1)
    distances = np.abs(offsets).max(axis=1)
    best = max(range(len(offsets)), key=lambda index: (votes[index], -distances[index]))
    dx, dy = np.median(offsets[near[best]], axis=0).tolist()
    needed = MIN_CAMERA_VOTES
    if max(abs(dx), abs(dy)) > MAX_SCROLL_PX:
        statics = min(
            sum(item.label not in MOBILE_CLASSES for item in frame) for frame in (previous, current)
        )
        needed = max(MIN_JUMP_VOTES, round(statics * JUMP_VOTE_SHARE))
    if votes[best] < needed:
        return None
    if abs(dx) <= CAMERA_SNAP_PX and abs(dy) <= CAMERA_SNAP_PX:
        return 0.0, 0.0
    return dx, dy


def _can_link(track: Track, detection: Detection, overlap: float) -> bool:
    if track.mobile != is_unit(detection) or overlap < LINK_OVERLAP:
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


def _majority(track: Track) -> str:
    return track.labels.most_common(1)[0][0]


def _stitch_distance(before: Track, after: Track) -> float | None:
    """How far apart two pieces are, or None if one can't be the other's continuation."""
    start = min(after.boxes)
    gap = start - before.last
    if not 0 < gap <= STITCH_MAX_GAP_FRAMES or _majority(before) != _majority(after):
        return None
    leaving = _centre(before.boxes[before.last])
    arriving = _centre(after.boxes[start])
    distance = float(np.hypot(arriving[0] - leaving[0], arriving[1] - leaving[1]))
    return distance if distance <= STITCH_BASE_PX + STITCH_PX_PER_FRAME * gap else None


def stitch(tracks: Sequence[Track]) -> list[Track]:
    """Join pieces of moving units across gaps too short for a unit to be swapped.

    Precision beats length here. A broken track only skips an object when scoring, while
    a wrong join blames a tracker for being right, so a close call stays broken.
    """
    pieces = [track for track in tracks if track.mobile]
    costs = np.full((len(pieces), len(pieces)), UNLINKABLE)
    for row, before in enumerate(pieces):
        for column, after in enumerate(pieces):
            if before is not after:
                distance = _stitch_distance(before, after)
                if distance is not None:
                    costs[row, column] = distance
    following: dict[int, int] = {}
    for row, column in zip(*linear_sum_assignment(costs), strict=True):
        if costs[row, column] < UNLINKABLE:
            rivals = (
                np.delete(costs[row], column).min(initial=UNLINKABLE),
                np.delete(costs[:, column], row).min(initial=UNLINKABLE),
            )
            if min(rivals) - costs[row, column] >= STITCH_MARGIN_PX:
                following[int(row)] = int(column)
    followers = set(following.values())
    joined = [track for track in tracks if not track.mobile]
    for index, head in enumerate(pieces):
        if index in followers:
            continue
        cursor = index
        while cursor in following:
            cursor = following[cursor]
            piece = pieces[cursor]
            head.boxes.update(piece.boxes)
            head.labels.update(piece.labels)
            head.contested.extend(piece.contested)
        joined.append(head)
    return joined


def split_at_contested(track: Track) -> list[Track]:
    """End a unit's track wherever two units could be confused, rather than guess."""
    if not track.mobile or not track.contested:
        return [track]
    pieces: list[Track] = []
    start = -1
    for index, end in enumerate([*sorted(set(track.contested)), max(track.boxes)]):
        boxes = {frame: box for frame, box in track.boxes.items() if start < frame <= end}
        if boxes:
            pieces.append(
                Track(f"{track.identifier}_{index}", True, boxes, Counter(track.labels), [end])
            )
        start = end
    return pieces


def link(detections: Sequence[Sequence[Detection]]) -> Linked:
    """Follow every object through the full-rate frames, in world coordinates."""
    tracks: list[Track] = []
    open_tracks: list[Track] = []
    shifts: list[Step] = []
    cuts: list[int] = []
    lost: set[int] = set()
    anchor = 0  # The last frame whose camera position is known.
    for frame, current in enumerate(detections):
        step = (0.0, 0.0) if frame == 0 else camera_step(detections[anchor], current)
        if step is None and frame - anchor <= MAX_LOST_FRAMES:
            shifts.append(shifts[anchor])
            lost.add(frame)
            continue
        if step is None:
            cuts.append(frame)
            open_tracks = []
            step = (0.0, 0.0)
            shift = shifts[anchor]
        else:
            base = shifts[anchor] if shifts else (0.0, 0.0)
            shift = (base[0] + step[0], base[1] + step[1])
        anchor = frame
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
                track = Track(f"track_{len(tracks)}", is_unit(detection))
                tracks.append(track)
                open_tracks.append(track)
            track.boxes[frame] = world[index]
            track.labels[detection.label] += 1
    whole = [piece for track in stitch(tracks) for piece in split_at_contested(track)]
    kept = tuple(track for track in whole if len(track.boxes) >= MIN_TRACK_FRAMES)
    return Linked(kept, tuple(shifts), tuple(cuts), frozenset(lost))


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


def _truth_label(track: Track) -> str:
    label = _majority(track)
    return NOT_A_UNIT if label in MOBILE_CLASSES and not track.mobile else label


def truth_at(linked: Linked, frame: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    if frame in linked.lost:
        return objects
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
                    "class_name": _truth_label(track),
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
        "lost_frames": len(linked.lost),
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


def _verified(linked: Linked, directory: Path) -> Linked:
    """Keep only the moving units a person checked, when a review exists.

    Automatic identity is reliable for static objects, but not for units in a crowd.
    """
    path = directory / VERIFIED
    if not path.exists():
        return linked
    approved = set(json.loads(path.read_text(encoding="utf-8"))["approved"])
    tracks = tuple(
        track for track in linked.tracks if not track.mobile or track.identifier in approved
    )
    return replace(linked, tracks=tracks)


def _sample_tracks(linked: Linked, count: int, seed: int) -> list[Track]:
    long_enough = [
        track
        for track in linked.tracks
        if track.mobile
        and len(set(track.boxes) & set(range(0, 10**6, SAMPLE_STEP))) >= STRIP_MIN_FRAMES
    ]
    random.Random(seed).shuffle(long_enough)
    return long_enough[:count]


def _strip_cell(directory: Path, linked: Linked, track: Track, frame: int) -> Image.Image:
    scale = STRIP_CELL_PX / STRIP_CROP_PX
    x0, y0, x1, y1 = _shifted(track.boxes[frame], linked.shifts[frame])
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(centre_x - STRIP_CROP_PX / 2), int(centre_y - STRIP_CROP_PX / 2)
    image = Image.open(directory / f"frame_{frame // SAMPLE_STEP:05d}.jpg").convert("RGB")
    cell = image.crop((left, top, left + STRIP_CROP_PX, top + STRIP_CROP_PX))
    cell = cell.resize((STRIP_CELL_PX, STRIP_CELL_PX), Image.Resampling.NEAREST)
    ImageDraw.Draw(cell).rectangle(
        [(x0 - left) * scale, (y0 - top) * scale, (x1 - left) * scale, (y1 - top) * scale],
        outline=(255, 40, 40),
        width=2,
    )
    return cell


def write_strips(directory: Path, count: int, seed: int) -> None:
    """Draw sampled tracks for review. No tracker output appears, so review can't favour one."""
    _, frames = _read_dense(directory)
    linked = link(frames)
    tracks = _sample_tracks(linked, count, seed)
    output = directory / "strips"
    output.mkdir(exist_ok=True)
    (output / "rows.json").write_text(
        json.dumps([track.identifier for track in tracks]), encoding="utf-8"
    )
    for sheet_index in range(0, len(tracks), STRIP_ROWS):
        rows = tracks[sheet_index : sheet_index + STRIP_ROWS]
        sheet = Image.new("RGB", (STRIP_CELL_PX * STRIP_COLUMNS, STRIP_CELL_PX * len(rows)))
        for row, track in enumerate(rows):
            sampled = sorted(frame for frame in track.boxes if frame % SAMPLE_STEP == 0)
            for column in range(STRIP_COLUMNS):
                frame = sampled[round(column * (len(sampled) - 1) / (STRIP_COLUMNS - 1))]
                sheet.paste(
                    _strip_cell(directory, linked, track, frame),
                    (column * STRIP_CELL_PX, row * STRIP_CELL_PX),
                )
        sheet.save(output / f"sheet_{sheet_index // STRIP_ROWS:02d}.jpg", quality=88)


def write_sequence(directory: Path) -> dict[str, object]:
    """Build the replay's files from cached detections, so truth rules can change freely."""
    times, frames = _read_dense(directory)
    linked = _verified(link(frames), directory)
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
            "same_spot_2s": scorecard.target_outcomes(frames, 2.0).same_spot_rate,
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
    print(
        "contender      1 s     2 s     5 s  mobile  static  wrong  same spot   IDF1  switches/1000"
    )
    for name, row in results.items():
        if name == "parity":
            continue
        print(
            f"{name:12s} {_percent(row['survival_1s'])} {_percent(row['survival_2s'])} "
            f"{_percent(row['survival_5s'])} {_percent(row['survival_2s_mobile'])} "
            f"{_percent(row['survival_2s_static'])} {row['wrong_2s'] * 100:5.2f}% "
            f"{_percent(row['same_spot_2s'])}     "
            f"{_percent(row['idf1'])} {row['switches']:8.0f}"
        )
    if "parity" in results:
        print(
            f"python and screentrack differ on {results['parity']['differing_frames']:.0f} frames"
        )


def pooled_outcome(
    runs: Sequence[Sequence[scorecard.ScoredFrame]], latency: float
) -> scorecard.TargetOutcome:
    outcomes = [scorecard.target_outcomes(frames, latency) for frames in runs]
    return scorecard.TargetOutcome(
        sum(item.survived for item in outcomes),
        sum(item.wrong for item in outcomes),
        sum(item.lost for item in outcomes),
        sum(item.same_spot for item in outcomes),
    )


def _mean_labels(runs: Sequence[Sequence[scorecard.ScoredFrame]]) -> tuple[float, float]:
    accuracy, flips = zip(*(scorecard.label_scores(frames) for frames in runs), strict=True)
    return float(np.mean(accuracy)), float(np.mean(flips))


def verdict(directories: Sequence[Path], name: str, rules: dict[str, object]) -> list[str]:
    """Apply the pre-agreed rules, pooling every clip, and describe each one."""
    latency = cast("float", rules["latency_seconds"])
    baseline = cast("str", rules["baseline"])
    label_baseline = cast("str", rules["label_baseline"])
    runs: dict[str, list[list[scorecard.ScoredFrame]]] = {}
    for entrant in {name, baseline, label_baseline}:
        runs[entrant] = [
            scorecard.score_frames(
                contenders.replay(contenders.contender(entrant), directory)[0],
                scorecard.read_truth(directory),
            )
            for directory in directories
        ]
    ours = pooled_outcome(runs[name], latency)
    theirs = pooled_outcome(runs[baseline], latency)
    accuracy, flips = _mean_labels(runs[name])
    plain_accuracy, plain_flips = _mean_labels(runs[label_baseline])
    checks = (
        ("survival", ours.survival >= theirs.survival, f"{ours.survival:.1%} against {theirs.survival:.1%}"),
        ("wrong object", ours.wrong_rate <= theirs.wrong_rate, f"{ours.wrong_rate:.2%} against {theirs.wrong_rate:.2%}"),
        ("labels", accuracy >= plain_accuracy and flips <= plain_flips, f"accuracy {accuracy:.2%} against {plain_accuracy:.2%}; flips {flips:.1f} against {plain_flips:.1f}"),
    )  # fmt: skip
    return [
        f"{'pass' if passed else 'FAIL'}  {label}: {detail}" for label, passed, detail in checks
    ]


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
    strips = commands.add_parser("strips")
    strips.add_argument("directory", type=Path)
    strips.add_argument("--count", type=int, default=24)
    strips.add_argument("--seed", type=int, default=SEED)
    judged = commands.add_parser("verdict")
    judged.add_argument("directories", type=Path, nargs="+")
    judged.add_argument("--name", default="screentrack")
    rate = commands.add_parser("score")
    rate.add_argument("directory", type=Path)
    rate.add_argument("--names", nargs="+", default=list(REPORT_NAMES))
    arguments = parser.parse_args()
    if arguments.command == "colour":
        _colour_command(arguments)
    elif arguments.command == "prepare":
        _prepare_command(arguments)
    elif arguments.command == "strips":
        write_strips(arguments.directory, arguments.count, arguments.seed)
    elif arguments.command == "verdict":
        rules = tomllib.loads(GATES.read_text(encoding="utf-8"))["recorded"]
        print("\n".join(verdict(arguments.directories, arguments.name, rules)))
    else:
        results = score(arguments.directory, arguments.names)
        _print_scores(results)
        (arguments.directory / "scores.json").write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
