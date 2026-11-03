"""Does object size alone explain the detector's trouble on Mac frames?

On the Mac, houses were about 1.7 times wider than in the VM frames the detector
was tested on. ``enlarge`` crops the VM test frames, so the server's letterbox
scales each crop up to the model input, and scores the detector three ways on
exactly the same objects:

- ``full``: the whole frame, as in the standard evaluation;
- ``masked``: only the crop is visible, pasted in place on a grey canvas, so
  objects keep their size but lose the view around them;
- ``enlarged``: the crop alone, so objects are ``scale`` times larger.

``masked`` against ``enlarged`` isolates object size. ``full`` against
``enlarged`` is the whole effect of zooming in.

Crops overlap by at least half. Each object is scored once, in the crop whose
core holds its centre, and only if that crop shows all of it. An object cut by
a crop edge is neither a hit nor a miss in that crop, and a guess on it is not
a false positive.

Run it with the main checkout's environment, which has the models and the test
set, from that checkout's root::

    .venv/bin/python <this worktree>/scripts/detector_scale_probe.py enlarge \\
        --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \\
        --data tmp/test_data_independent_20261030 --output <dir>/v9.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from detection.inference.postprocess import iou

if TYPE_CHECKING:
    from collections.abc import Sequence

    from detection_server.app import ModelState
    from PIL import Image

Box = tuple[float, float, float, float]
Variant = Literal["full", "masked", "enlarged"]

MATCH_IOU = 0.5
IGNORE_COVER = 0.5
PAD_COLOR = (114, 114, 114)
DEFAULT_SCALES = (1.25, 1.5, 1.7, 2.0)
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20261103
REPORTED_CLASS_MINIMUM = 20
_EVERYWHERE: Box = (-math.inf, -math.inf, math.inf, math.inf)


@dataclass(frozen=True, slots=True)
class Truth:
    class_id: int
    box: Box


@dataclass(frozen=True, slots=True)
class Guess:
    class_id: int
    box: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class Tile:
    """A crop in frame pixels, and the core whose objects it scores."""

    left: int
    top: int
    right: int
    bottom: int
    core: Box


@dataclass(frozen=True, slots=True)
class Shown:
    """A truth as one crop shows it: the visible box, and whether it counts there."""

    class_id: int
    box: Box
    scored: bool


@dataclass(slots=True)
class Tally:
    hits: Counter[int] = field(default_factory=Counter)
    false: Counter[int] = field(default_factory=Counter)
    missed: Counter[int] = field(default_factory=Counter)

    def add(self, other: Tally) -> None:
        self.hits.update(other.hits)
        self.false.update(other.false)
        self.missed.update(other.missed)

    def micro(self) -> tuple[int, int, int]:
        return self.hits.total(), self.false.total(), self.missed.total()


def f1(hits: int, false: int, missed: int) -> float:
    return 2 * hits / (2 * hits + false + missed) if hits else 0.0


def crops_per_axis(scale: float) -> int:
    """Enough crops along an axis that neighbours overlap by at least half a crop."""
    return max(2, math.ceil(2 * scale - 1 - 1e-9))


def _starts(length: int, crop: int, count: int) -> list[int]:
    return [round(index * (length - crop) / (count - 1)) for index in range(count)]


def _cores(starts: list[int], crop: int) -> list[tuple[float, float]]:
    centres = [start + crop / 2 for start in starts]
    bounds = [-math.inf, *((a + b) / 2 for a, b in pairwise(centres)), math.inf]
    return list(pairwise(bounds))


def tile_frame(width: int, height: int, scale: float) -> list[Tile]:
    """Overlapping crops of 1/scale of the frame whose cores split it between them."""
    if scale == 1:
        return [Tile(0, 0, width, height, _EVERYWHERE)]
    crop_width, crop_height = round(width / scale), round(height / scale)
    count = crops_per_axis(scale)
    xs, ys = _starts(width, crop_width, count), _starts(height, crop_height, count)
    return [
        Tile(x, y, x + crop_width, y + crop_height, (core_left, core_top, core_right, core_bottom))
        for y, (core_top, core_bottom) in zip(ys, _cores(ys, crop_height), strict=True)
        for x, (core_left, core_right) in zip(xs, _cores(xs, crop_width), strict=True)
    ]


def _centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _within(point: tuple[float, float], area: Box) -> bool:
    x, y = point
    return area[0] <= x < area[2] and area[1] <= y < area[3]


def _cover(box: Box, area: Box) -> float:
    """Share of ``box`` that lies inside ``area``."""
    width = min(box[2], area[2]) - max(box[0], area[0])
    height = min(box[3], area[3]) - max(box[1], area[1])
    size = (box[2] - box[0]) * (box[3] - box[1])
    return width * height / size if width > 0 and height > 0 and size > 0 else 0.0


def shown_in(truths: Sequence[Truth], tile: Tile, width: int, height: int) -> list[Shown]:
    """What the crop shows of each truth.

    Only the crop's inner edges cut boxes: labels already stop at the frame.
    """
    left = tile.left if tile.left > 0 else -math.inf
    top = tile.top if tile.top > 0 else -math.inf
    right = tile.right if tile.right < width else math.inf
    bottom = tile.bottom if tile.bottom < height else math.inf
    shown: list[Shown] = []
    for truth in truths:
        x0, y0, x1, y1 = truth.box
        box = (max(x0, left), max(y0, top), min(x1, right), min(y1, bottom))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        whole = box == truth.box
        shown.append(Shown(truth.class_id, box, whole and _within(_centre(truth.box), tile.core)))
    return shown


def scored_somewhere(
    truths: Sequence[Truth], tiles: Sequence[Tile], width: int, height: int
) -> list[bool]:
    """Which truths some crop scores. A large object cut by its own crop is never scored."""
    return [
        any(shown.scored for tile in tiles for shown in shown_in([truth], tile, width, height))
        for truth in truths
    ]


def score(guesses: Sequence[Guess], shown: Sequence[Shown], core: Box) -> Tally:
    """Greedy matching as in the evaluator's scorer, plus ignored truths.

    A guess matched to a scored truth is a hit. A guess matched to an ignored
    truth, lying mostly on one, or centred outside the core counts nowhere.
    """
    tally = Tally()
    used = [False] * len(shown)
    for guess in sorted(guesses, key=lambda item: -item.confidence):
        best, best_iou = -1, MATCH_IOU
        for index, truth in enumerate(shown):
            if used[index] or truth.class_id != guess.class_id:
                continue
            overlap = iou(guess.box, truth.box)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            used[best] = True
            if shown[best].scored:
                tally.hits[guess.class_id] += 1
        elif _within(_centre(guess.box), core) and not any(
            not truth.scored and _cover(guess.box, truth.box) >= IGNORE_COVER for truth in shown
        ):
            tally.false[guess.class_id] += 1
    for truth, hit in zip(shown, used, strict=True):
        if truth.scored and not hit:
            tally.missed[truth.class_id] += 1
    return tally


def bootstrap_gap(
    before: Sequence[Tally],
    after: Sequence[Tally],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """90% range of F1(after) - F1(before), resampling whole frames, paired."""
    pairs = [(a.micro(), b.micro()) for a, b in zip(before, after, strict=True)]
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(rounds):
        picks = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        first = [sum(pick[0][part] for pick in picks) for part in range(3)]
        second = [sum(pick[1][part] for pick in picks) for part in range(3)]
        gaps.append(f1(*second) - f1(*first))
    gaps.sort()
    return gaps[math.floor(0.05 * (rounds - 1))], gaps[math.ceil(0.95 * (rounds - 1))]


def _shift(guesses: Sequence[Guess], dx: int, dy: int) -> list[Guess]:
    return [
        Guess(
            g.class_id, (g.box[0] + dx, g.box[1] + dy, g.box[2] + dx, g.box[3] + dy), g.confidence
        )
        for g in guesses
    ]


def _guesses(state: ModelState, image: Image.Image) -> list[Guess]:
    from detection_server.evaluate import predict

    return [Guess(p.class_id, p.box, p.confidence) for p in predict(state, image)]


def _view(image: Image.Image, tile: Tile, variant: Variant) -> Image.Image:
    from PIL import Image as PILImage

    crop = image.crop((tile.left, tile.top, tile.right, tile.bottom))
    if variant == "enlarged":
        return crop
    canvas = PILImage.new("RGB", image.size, PAD_COLOR)
    canvas.paste(crop, (tile.left, tile.top))
    return canvas


@dataclass(slots=True)
class ScaleResult:
    frames: dict[Variant, list[Tally]] = field(
        default_factory=lambda: {"full": [], "masked": [], "enlarged": []}
    )
    truths: int = 0
    unscored: int = 0


def _measure(
    state: ModelState, frames: Sequence[tuple[Path, Path]], scales: Sequence[float]
) -> tuple[list[Tally], dict[float, ScaleResult]]:
    from detection.labeling.class_mapping import get_classes_for_cvat
    from detection_server.evaluate import load_ground_truth
    from PIL import Image as PILImage

    class_count = len(get_classes_for_cvat())
    baseline: list[Tally] = []
    results = {scale: ScaleResult() for scale in scales}
    for image_path, label_path in frames:
        with PILImage.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        truths = [
            Truth(item.class_id, item.box)
            for item in load_ground_truth(label_path, image.size, class_count)
        ]
        full = _guesses(state, image)
        baseline.append(score(full, [Shown(t.class_id, t.box, True) for t in truths], _EVERYWHERE))
        for scale, result in results.items():
            tiles = tile_frame(width, height, scale)
            scored = scored_somewhere(truths, tiles, width, height)
            result.truths += len(truths)
            result.unscored += scored.count(False)
            result.frames["full"].append(
                score(
                    full,
                    [Shown(t.class_id, t.box, s) for t, s in zip(truths, scored, strict=True)],
                    _EVERYWHERE,
                )
            )
            for variant in ("masked", "enlarged"):
                frame = Tally()
                for tile in tiles:
                    guesses = _guesses(state, _view(image, tile, variant))
                    if variant == "enlarged":
                        guesses = _shift(guesses, tile.left, tile.top)
                    frame.add(score(guesses, shown_in(truths, tile, width, height), tile.core))
                result.frames[variant].append(frame)
    return baseline, results


def _summary(frames: Sequence[Tally]) -> dict[str, float | int]:
    total = Tally()
    for frame in frames:
        total.add(frame)
    hits, false, missed = total.micro()
    return {
        "hits": hits,
        "false": false,
        "missed": missed,
        "precision": hits / (hits + false) if hits + false else 0.0,
        "recall": hits / (hits + missed) if hits + missed else 0.0,
        "f1": f1(hits, false, missed),
    }


def _class_recall(
    frames: Sequence[Tally], names: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    total = Tally()
    for frame in frames:
        total.add(frame)
    return {
        names[class_id]: {
            "truths": total.hits[class_id] + total.missed[class_id],
            "recall": total.hits[class_id] / (total.hits[class_id] + total.missed[class_id]),
            "false": total.false[class_id],
        }
        for class_id in sorted(set(total.hits) | set(total.missed))
        if total.hits[class_id] + total.missed[class_id] >= REPORTED_CLASS_MINIMUM
    }


def _report(
    model: Path,
    data: Path,
    frames: Sequence[tuple[Path, Path]],
    baseline: list[Tally],
    results: dict[float, ScaleResult],
) -> dict[str, object]:
    from importlib.metadata import version
    from platform import python_version

    import detection_server.evaluate as evaluator
    from detection.labeling.class_mapping import get_classes_for_cvat

    names = get_classes_for_cvat()
    cases = [evaluator.Case(image=image, label=label) for image, label in frames]
    scales: dict[str, object] = {}
    for scale, result in results.items():
        full, masked, enlarged = (result.frames[v] for v in ("full", "masked", "enlarged"))
        scales[str(scale)] = {
            "crops_per_frame": crops_per_axis(scale) ** 2,
            "truths": result.truths,
            "unscored_truths": result.unscored,
            "full": _summary(full),
            "masked": _summary(masked),
            "enlarged": _summary(enlarged),
            "f1_gap_90": {
                "size (enlarged - masked)": bootstrap_gap(masked, enlarged),
                "crop (masked - full)": bootstrap_gap(full, masked),
                "zoom (enlarged - full)": bootstrap_gap(full, enlarged),
            },
            "class_recall": {
                "full": _class_recall(full, names),
                "masked": _class_recall(masked, names),
                "enlarged": _class_recall(enlarged, names),
            },
        }
    return {
        "model": str(model),
        "model_sha256": evaluator.sha256_file(model),
        "data": str(data),
        "dataset_manifest_sha256": evaluator.manifest_sha256(cases),
        "evaluator_sha256": evaluator.sha256_file(Path(inspect.getfile(evaluator))),
        "runtime": {
            "python": python_version(),
            "onnxruntime": version("onnxruntime"),
            "pillow": version("Pillow"),
        },
        "frames": len(frames),
        "matching_iou": MATCH_IOU,
        "ignore_cover": IGNORE_COVER,
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED, "unit": "frame"},
        "baseline": _summary(baseline),
        "scales": scales,
    }


def _print(report: dict[str, object]) -> None:
    baseline = report["baseline"]
    assert isinstance(baseline, dict)
    print(f"{report['model']}: {report['frames']} frames, baseline F1 {baseline['f1']:.4f}")
    print(f"{'scale':>6} {'scored':>7} {'full':>7} {'masked':>7} {'enlarged':>9}   size gap (90%)")
    scales = report["scales"]
    assert isinstance(scales, dict)
    for scale, row in scales.items():
        scored = row["truths"] - row["unscored_truths"]
        low, high = row["f1_gap_90"]["size (enlarged - masked)"]
        print(
            f"{scale:>6} {scored:>7} {row['full']['f1']:>7.3f} {row['masked']['f1']:>7.3f}"
            f" {row['enlarged']['f1']:>9.3f}   {row['enlarged']['f1'] - row['masked']['f1']:+.3f}"
            f" [{low:+.3f}, {high:+.3f}]"
        )


def _frames(data: Path, split: str) -> list[tuple[Path, Path]]:
    images = sorted(
        path
        for path in (data / split / "images").glob("real_*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No real images in {data / split / 'images'}")
    frames = [(image, data / split / "labels" / f"{image.stem}.txt") for image in images]
    missing = [label for _, label in frames if not label.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} labels, first: {missing[0]}")
    return frames


class _Args(argparse.Namespace):
    command: str
    model: Path
    data: Path
    split: str
    output: Path
    scales: list[float]


def _enlarge(args: _Args) -> int:
    from detection.labeling.class_mapping import model_classes_match_schema_prefix
    from detection_server.evaluate import load_cpu_model

    if any(scale <= 1 for scale in args.scales):
        raise ValueError("Scales must be above 1; the baseline is always measured")
    frames = _frames(args.data, args.split)
    state = load_cpu_model(args.model)
    if not model_classes_match_schema_prefix(state.class_names):
        raise ValueError("Detector server and training class schemas disagree")
    baseline, results = _measure(state, frames, args.scales)
    report = _report(args.model, args.data, frames, baseline, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _print(report)
    print(f"Wrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    enlarge = commands.add_parser(
        "enlarge", help="score the detector on enlarged crops of labelled frames"
    )
    enlarge.add_argument("--model", required=True, type=Path)
    enlarge.add_argument("--data", required=True, type=Path)
    enlarge.add_argument("--split", default="test")
    enlarge.add_argument("--output", required=True, type=Path)
    enlarge.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    args = parser.parse_args(namespace=_Args())
    return _enlarge(args)


if __name__ == "__main__":
    raise SystemExit(main())
