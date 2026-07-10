"""Villager job inference + motion-robust selection.

The YOLO model emits one undifferentiated ``villager`` class, so "which villagers
are on wood" is not something detection answers directly. This module infers each
villager's job from geometry — proximity to gatherable resources and drop-off camps
— and exposes:

  - ``infer_jobs`` / ``VillagerRoleModel``: per-villager job tags (single-frame and
    smoothed-over-frames), plus ``job_counts`` for LLM context.
  - ``select_worker``: pick a worker of a given job to command. Selecting a
    *moving* villager is the hard part; when per-track velocities are available
    (from the Kalman tracker) we prefer the most stationary worker — a villager
    chopping/mining sits near-still and is trivial to click.

Pure functions over ``DetectedEntity``-or-dict inputs (via ``entity_utils``); no
detector, executor, or pyautogui coupling — the reassignment tool wires those in.
"""

from __future__ import annotations

import math
from typing import Literal

from .entity_utils import (
    CAMP_CLASS_BY_KIND,
    CLASSES_BY_KIND,
    RESOURCE_KINDS,
    EntityAttrs,
    ResourceKind,
    dist,
    iter_attrs,
)

# A villager's inferred job: a gatherable resource kind, or idle/unknown.
# (Implicit alias form; the repo targets Python 3.11, pre-PEP-695.)
VillagerJob = ResourceKind | Literal["idle"]

# A villager within this many px of a resource/camp is counted as working it. Tuned
# for a zoomed-in economy view; a villager standing on a tree/mine is well inside.
JOB_RADIUS = 140.0
IDLE: VillagerJob = "idle"

# class_name → job kind, built once from the shared taxonomy. Gather classes and
# camps both signal the job; the taxonomy keeps camps unambiguous (see entity_utils).
_CLASS_TO_KIND: dict[str, ResourceKind] = {
    cls: kind
    for mapping in (CLASSES_BY_KIND, CAMP_CLASS_BY_KIND)
    for kind, classes in mapping.items()
    for cls in classes
}


def _job_anchors(entities: list[object]) -> list[tuple[ResourceKind, tuple[float, float]]]:
    """(kind, center) for every resource/camp on screen — the job evidence."""
    anchors: list[tuple[ResourceKind, tuple[float, float]]] = []
    for a in iter_attrs(entities):
        kind = _CLASS_TO_KIND.get(a.class_name)
        if kind is not None:
            anchors.append((kind, a.center))
    return anchors


def classify_job(
    center: tuple[float, float],
    anchors: list[tuple[ResourceKind, tuple[float, float]]],
    radius: float = JOB_RADIUS,
) -> VillagerJob:
    """Job of a villager at `center`: the kind of the nearest anchor within radius.

    Returns ``IDLE`` when no resource/camp is close enough (the villager is walking
    or genuinely idle). `anchors` is precomputed once per frame via `_job_anchors`.
    """
    best_kind: VillagerJob = IDLE
    best_d = radius
    for kind, pos in anchors:
        d = dist(center, pos)
        if d < best_d:
            best_kind, best_d = kind, d
    return best_kind


def infer_jobs(entities: list[object]) -> dict[str, VillagerJob]:
    """Single-frame ``{villager_id: job}`` for every detected villager."""
    anchors = _job_anchors(entities)
    jobs: dict[str, VillagerJob] = {}
    for a in iter_attrs(entities):
        if a.class_name == "villager":
            jobs[a.entity_id] = classify_job(a.center, anchors)
    return jobs


def job_counts(jobs: dict[str, VillagerJob]) -> dict[VillagerJob, int]:
    """Count villagers per job (all kinds + idle present, even at zero)."""
    counts: dict[VillagerJob, int] = dict.fromkeys((*RESOURCE_KINDS, IDLE), 0)
    for job in jobs.values():
        counts[job] = counts.get(job, 0) + 1
    return counts


class VillagerRoleModel:
    """Smooths per-villager jobs over frames to resist single-frame noise.

    A villager walking from trees toward the mill flickers between ``wood`` and
    ``food`` frame-to-frame; a short majority vote per tracked id (stable across
    frames thanks to the Kalman tracker) keeps the label steady. Feed each frame's
    entities to ``update``; read ``counts`` for context.
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window
        # {villager_id: recent job labels (most recent last)}
        self._history: dict[str, list[VillagerJob]] = {}
        self._smoothed: dict[str, VillagerJob] = {}

    def update(self, entities: list[object]) -> dict[str, VillagerJob]:
        raw = infer_jobs(entities)
        alive = set(raw)
        # Drop tracks that vanished so counts don't count ghosts.
        for gone in [vid for vid in self._history if vid not in alive]:
            del self._history[gone]
            self._smoothed.pop(gone, None)
        for vid, job in raw.items():
            hist = self._history.setdefault(vid, [])
            hist.append(job)
            if len(hist) > self.window:
                del hist[0]
            self._smoothed[vid] = max(set(hist), key=hist.count)  # majority vote
        return dict(self._smoothed)

    def counts(self) -> dict[VillagerJob, int]:
        return job_counts(self._smoothed)


# ---------------------------------------------------------------------------
# Selection — pick a worker of a given job to command
# ---------------------------------------------------------------------------


def _speed(velocities: dict[str, tuple[float, float]] | None, vid: str) -> float:
    if not velocities or vid not in velocities:
        return math.inf  # unknown velocity sorts last so known-stationary wins
    vx, vy = velocities[vid]
    return math.hypot(vx, vy)


def select_worker(
    entities: list[object],
    job: ResourceKind,
    velocities: dict[str, tuple[float, float]] | None = None,
) -> tuple[int, int] | None:
    """Click point of a villager working `job`, or None when none is visible.

    Prefers the most *stationary* worker when per-track velocities are supplied
    (easiest to click); otherwise the one nearest its resource/camp (most solidly
    "on the job").
    """
    all_anchors = _job_anchors(entities)
    job_anchors = [pos for kind, pos in all_anchors if kind == job]
    candidates = [
        a
        for a in iter_attrs(entities)
        if a.class_name == "villager" and classify_job(a.center, all_anchors) == job
    ]
    if not candidates:
        return None

    def rank(a: EntityAttrs) -> tuple[float, float]:
        speed = _speed(velocities, a.entity_id)
        nearest_anchor = min((dist(a.center, p) for p in job_anchors), default=0.0)
        return (speed, nearest_anchor)  # stationary first, then closest to the job

    best = min(candidates, key=rank)
    return (int(best.center[0]), int(best.center[1]))
