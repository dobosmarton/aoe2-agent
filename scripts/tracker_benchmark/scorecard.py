"""Scoring against ground truth, built from what the agent does with IDs and labels.

Identity scores only count objects the detector found, because a missed box isn't the
tracker's fault. That is what lets `oracle` reach exactly 100% and prove the metrics work.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from gameplay_agent.camera_motion import MOBILE_CLASSES
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.sequence_replay import _overlap
from gameplay_agent.villager_roles import infer_jobs, job_counts
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from contenders import FrameRecord

Box = tuple[float, float, float, float]

MATCH_OVERLAP = 0.5
SUMMARY_LIMIT = 20  # build_entity_summary's default: what the LLM actually sees.
STABLE_FRAMES = 5
BOOTSTRAP_ROUNDS = 2000
AGREEMENT_TOLERANCE = 1e-9
PER_THOUSAND = 1000.0


@dataclass(frozen=True, slots=True)
class TruthObject:
    identifier: str
    label: str
    box: Box


@dataclass(frozen=True, slots=True)
class ScoredFrame:
    """One frame with each prediction paired to the ground-truth object it found."""

    at: float
    predictions: tuple[EntitySnapshot, ...]
    truths: tuple[TruthObject, ...]
    found: dict[str, EntitySnapshot]

    @property
    def owner_of(self) -> dict[str, str]:
        return {prediction.id: identifier for identifier, prediction in self.found.items()}


def match_frame(
    predictions: Sequence[EntitySnapshot], truths: Sequence[TruthObject]
) -> dict[str, EntitySnapshot]:
    """Pair predictions and truth one-to-one, best overlap first."""
    pairs = sorted(
        (
            (_overlap(prediction.bbox or (0, 0, 0, 0), truth.box), index, truth.identifier)
            for index, prediction in enumerate(predictions)
            for truth in truths
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    found: dict[str, EntitySnapshot] = {}
    used: set[int] = set()
    for overlap, index, identifier in pairs:
        if overlap < MATCH_OVERLAP:
            break
        if index not in used and identifier not in found:
            found[identifier] = predictions[index]
            used.add(index)
    return found


def read_truth(directory: Path) -> dict[int, tuple[TruthObject, ...]]:
    truth: dict[int, tuple[TruthObject, ...]] = {}
    for line in (directory / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        truth[row["capture_id"]] = tuple(
            TruthObject(item["track_id"], item["class_name"], tuple(item["bbox"]))
            for item in row["objects"]
        )
    return truth


def score_frames(
    records: Iterable[FrameRecord], truth: dict[int, tuple[TruthObject, ...]]
) -> list[ScoredFrame]:
    frames: list[ScoredFrame] = []
    for record in records:
        objects = truth.get(record.capture_id, ())
        frames.append(
            ScoredFrame(
                record.captured_at,
                record.entities,
                objects,
                match_frame(record.entities, objects),
            )
        )
    return frames


def _is_mobile(label: str) -> bool:
    return label in MOBILE_CLASSES


def _truth_labels(frames: Sequence[ScoredFrame]) -> dict[str, str]:
    return {truth.identifier: truth.label for frame in frames for truth in frame.truths}


def _included(identifier: str, labels: dict[str, str], mobile: bool | None) -> bool:
    return mobile is None or _is_mobile(labels[identifier]) == mobile


def found_count(frames: Sequence[ScoredFrame], mobile: bool | None = None) -> int:
    """Objects found across all frames, used to weight scenes fairly."""
    labels = _truth_labels(frames)
    return sum(
        _included(identifier, labels, mobile) for frame in frames for identifier in frame.found
    )


def idf1(frames: Sequence[ScoredFrame], mobile: bool | None = None) -> float:
    """Share of found objects carrying their object's single best ID."""
    labels = _truth_labels(frames)
    counts: Counter[tuple[str, str]] = Counter()
    for frame in frames:
        for identifier, prediction in frame.found.items():
            if _included(identifier, labels, mobile):
                counts[(identifier, prediction.id)] += 1
    total = sum(counts.values())
    if not total:
        return 0.0
    truths = sorted({truth for truth, _ in counts})
    predictions = sorted({prediction for _, prediction in counts})
    grid = np.zeros((len(truths), len(predictions)))
    rows = {name: index for index, name in enumerate(truths)}
    columns = {name: index for index, name in enumerate(predictions)}
    for (truth, prediction), count in counts.items():
        grid[rows[truth], columns[prediction]] = count
    row_index, column_index = linear_sum_assignment(grid, maximize=True)
    return float(grid[row_index, column_index].sum()) / total


def id_switches(frames: Sequence[ScoredFrame], mobile: bool | None = None) -> float:
    """Identity changes per 1,000 object-frames, for objects found in both frames."""
    labels = _truth_labels(frames)
    switches = pairs = 0
    for previous, current in itertools.pairwise(frames):
        for identifier, prediction in current.found.items():
            if identifier in previous.found and _included(identifier, labels, mobile):
                pairs += 1
                switches += previous.found[identifier].id != prediction.id
    return switches / pairs * PER_THOUSAND if pairs else 0.0


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """What happens when the agent acts on an ID it saw `latency` seconds ago."""

    survived: int
    wrong: int
    lost: int

    @property
    def total(self) -> int:
        return self.survived + self.wrong + self.lost

    @property
    def survival(self) -> float:
        return self.survived / self.total if self.total else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.total if self.total else 0.0


def target_outcomes(
    frames: Sequence[ScoredFrame], latency_seconds: float, mobile: bool | None = None
) -> TargetOutcome:
    """Apply `_resolve_target_id`'s exact-ID rule to what the summary showed earlier.

    Only objects the detector still finds later count, so `oracle` scores 100%.
    """
    labels = _truth_labels(frames)
    survived = wrong = lost = 0
    for index, frame in enumerate(frames):
        later = next(
            (item for item in frames[index + 1 :] if item.at >= frame.at + latency_seconds),
            None,
        )
        if later is None:
            break
        owners = frame.owner_of
        later_owners = later.owner_of
        for prediction in frame.predictions[:SUMMARY_LIMIT]:
            identifier = owners.get(prediction.id)
            if (
                identifier is None
                or identifier not in later.found
                or not _included(identifier, labels, mobile)
            ):
                continue
            resolved = later_owners.get(prediction.id)
            if resolved == identifier:
                survived += 1
            elif resolved is None:
                lost += 1
            else:
                wrong += 1
    return TargetOutcome(survived, wrong, lost)


def label_scores(frames: Sequence[ScoredFrame]) -> tuple[float, float]:
    """Label accuracy, and label flips per 1,000 object-frames."""
    labels = _truth_labels(frames)
    correct = total = flips = pairs = 0
    for frame in frames:
        for identifier, prediction in frame.found.items():
            total += 1
            correct += prediction.class_name == labels[identifier]
    for previous, current in itertools.pairwise(frames):
        for identifier, prediction in current.found.items():
            if identifier in previous.found:
                pairs += 1
                flips += previous.found[identifier].class_name != prediction.class_name
    accuracy = correct / total if total else 0.0
    return accuracy, flips / pairs * PER_THOUSAND if pairs else 0.0


def _as_snapshot(truth: TruthObject) -> EntitySnapshot:
    x0, y0, x1, y1 = truth.box
    return EntitySnapshot(
        truth.identifier, truth.label, truth.box, ((x0 + x1) / 2, (y0 + y1) / 2), 1.0
    )


def job_error(frames: Sequence[ScoredFrame]) -> float:
    """Mean per-frame error in the real `job_counts`, against ground truth."""
    errors: list[int] = []
    for frame in frames:
        seen = job_counts(infer_jobs(list(frame.predictions)))
        actual = job_counts(infer_jobs([_as_snapshot(item) for item in frame.truths]))
        errors.append(sum(abs(seen[kind] - actual[kind]) for kind in actual))
    return float(np.mean(errors)) if errors else 0.0


def recovery_seconds(
    frames: Sequence[ScoredFrame], identifier: str, disturbed_at: float
) -> float | None:
    """Time from a disturbance until one ID holds for `STABLE_FRAMES`, or None if never."""
    after = [frame for frame in frames if frame.at >= disturbed_at and identifier in frame.found]
    for index in range(len(after) - STABLE_FRAMES + 1):
        window = {frame.found[identifier].id for frame in after[index : index + STABLE_FRAMES]}
        if len(window) == 1:
            return after[index].at - disturbed_at
    return None


def first_difference(left: Sequence[EntitySnapshot], right: Sequence[EntitySnapshot]) -> str | None:
    """The first decision two contenders disagree on, for the parity gate."""
    if len(left) != len(right):
        return f"{len(left)} versus {len(right)} detections"
    for index, (ours, theirs) in enumerate(zip(left, right, strict=True)):
        for name in ("id", "class_name", "raw_class_name"):
            if getattr(ours, name) != getattr(theirs, name):
                return f"detection {index} {name}"
        scores = (ours.class_agreement, theirs.class_agreement)
        if (scores[0] is None) != (scores[1] is None):
            return f"detection {index} agreement presence"
        if (
            scores[0] is not None
            and scores[1] is not None
            and abs(scores[0] - scores[1]) > AGREEMENT_TOLERANCE
        ):
            return f"detection {index} agreement value"
    return None


def percentile(values: Sequence[float], share: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), share, method="higher"))


def bootstrap_ratio(
    slower: Sequence[float], faster: Sequence[float], *, share: float = 0.95, seed: int = 11
) -> tuple[float, float, float]:
    """Paired bootstrap of `slower_p95 / faster_p95`: low bound, estimate, high bound."""
    left = np.asarray(slower, dtype=np.float64)
    right = np.asarray(faster, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("a paired ratio needs the same frames on both sides")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, left.size, (BOOTSTRAP_ROUNDS, left.size))
    ratios = np.quantile(left[picks], share, axis=1) / np.quantile(right[picks], share, axis=1)
    low, high = np.quantile(ratios, (0.025, 0.975))
    return float(low), float(np.quantile(left, share) / np.quantile(right, share)), float(high)
