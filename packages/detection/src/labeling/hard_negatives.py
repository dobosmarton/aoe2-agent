"""Surface cavalry-line confusions from triage for targeted re-labeling.

The light-cav <-> heavy-cav confusion is the dominant driver of the rare-class
accuracy gap. `active_learning.triage` already runs the model over every raw
screenshot at a low confidence floor and keeps each detection; this module
filters those detections to the confusable cavalry classes at low confidence, so
a human can re-annotate exactly the crops the model is unsure about (fed back
through `active_learning.prepare_batch`).

Usage:
    python -m detection.labeling.hard_negatives --max-conf 0.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .active_learning import triage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .active_learning import DetectionRecord, TriageItem

# Cavalry lines the model most often confuses (classes.yaml names).
CAVALRY_CONFUSION_CLASSES: frozenset[str] = frozenset(
    {"scout_line", "knight_line", "camel_line", "battle_elephant", "cavalry_archer"}
)


@dataclass(frozen=True, slots=True)
class ConfusableHit:
    """A low-confidence detection of a confusable class on a specific image."""

    image: str
    detection: DetectionRecord


def find_confusable_detections(
    items: Sequence[TriageItem],
    focus_classes: frozenset[str] = CAVALRY_CONFUSION_CLASSES,
    max_confidence: float = 0.5,
) -> list[ConfusableHit]:
    """Return detections of `focus_classes` below `max_confidence`, least-sure first.

    These are the hard negatives worth re-annotating: the model predicted a
    confusable cavalry class but was not confident, so the ground truth is likely
    one of the sibling classes.
    """
    hits = [
        ConfusableHit(image=item.path, detection=detection)
        for item in items
        for detection in item.detections
        if detection.class_name in focus_classes and detection.confidence < max_confidence
    ]
    hits.sort(key=lambda hit: hit.detection.confidence)
    return hits


def _summarise_by_image(hits: Sequence[ConfusableHit]) -> dict[str, int]:
    """Count confusable hits per image path."""
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.image] = counts.get(hit.image, 0) + 1
    return counts


class _HardNegativeArgs(argparse.Namespace):
    model: str | None
    max_conf: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surface cavalry confusions from triage for re-labeling",
    )
    parser.add_argument("--model", type=str, default=None, help="Detection model path")
    parser.add_argument("--max-conf", type=float, default=0.5, help="Confidence ceiling")
    args = parser.parse_args(namespace=_HardNegativeArgs())

    items = triage(Path(args.model)) if args.model else triage()
    hits = find_confusable_detections(items, max_confidence=args.max_conf)

    print(f"\n{len(hits)} confusable cavalry detections below {args.max_conf} confidence:")
    for image, count in sorted(_summarise_by_image(hits).items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {Path(image).name}")
    print("\nRe-label these via: python -m detection.labeling.active_learning prepare")


if __name__ == "__main__":
    main()
