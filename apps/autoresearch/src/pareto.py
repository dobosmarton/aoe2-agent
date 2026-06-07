"""Pareto frontier over the 5 score components (A2).

Keeps prompt-edit candidates that are non-dominated across
(survival, population, age, economy, action_success) so a candidate that is
strong on one axis isn't discarded just because its weighted composite isn't
the best. Pure functions + JSON persistence; no external dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import structlog

log = structlog.stdlib.get_logger()

_FRONTIER_PATH = Path(__file__).parent.parent / "experiments" / "pareto_frontier.json"
AXES: tuple[str, ...] = ("survival", "population", "age", "economy", "action_success")

Vector = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ParetoEntry:
    """A candidate prompt edit and its per-component score vector."""

    candidate_id: str
    description: str
    change: dict[str, str]
    vector: Vector


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True if a is >= b on every axis and strictly greater on at least one."""
    pairs = list(zip(a, b, strict=True))
    return all(x >= y for x, y in pairs) and any(x > y for x, y in pairs)


def update_frontier(frontier: list[ParetoEntry], candidate: ParetoEntry) -> list[ParetoEntry]:
    """Insert candidate: reject if dominated, else drop entries it dominates."""
    if any(dominates(e.vector, candidate.vector) for e in frontier):
        return list(frontier)
    kept = [e for e in frontier if not dominates(candidate.vector, e.vector)]
    kept.append(candidate)
    return kept


def _entry_to_dict(e: ParetoEntry) -> dict[str, object]:
    return {
        "candidate_id": e.candidate_id,
        "description": e.description,
        "change": e.change,
        "vector": list(e.vector),
    }


def _entry_from_dict(d: dict[str, object]) -> ParetoEntry | None:
    vec = d.get("vector")
    change = d.get("change")
    if not isinstance(vec, list) or len(vec) != 5 or not isinstance(change, dict):
        return None
    vector = cast("Vector", tuple(float(x) for x in vec))
    change_str = {str(k): str(v) for k, v in change.items()}
    return ParetoEntry(
        candidate_id=str(d.get("candidate_id", "")),
        description=str(d.get("description", "")),
        change=change_str,
        vector=vector,
    )


def load_frontier(path: Path | None = None) -> list[ParetoEntry]:
    """Load the persisted frontier (empty list if absent or unreadable)."""
    p = path or _FRONTIER_PATH
    if not p.exists():
        return []
    try:
        raw = cast("object", json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    entries = [_entry_from_dict(item) for item in raw if isinstance(item, dict)]
    return [e for e in entries if e is not None]


def save_frontier(frontier: list[ParetoEntry], path: Path | None = None) -> None:
    """Persist the frontier as JSON next to the experiment ledger."""
    p = path or _FRONTIER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([_entry_to_dict(e) for e in frontier], indent=2))
    log.info("pareto_frontier_saved", size=len(frontier), path=str(p))
