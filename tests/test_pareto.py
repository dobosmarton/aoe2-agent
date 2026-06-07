"""Unit tests for the A2 Pareto frontier (autoresearch/pareto.py). Pure + JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autoresearch.pareto import (
    ParetoEntry,
    dominates,
    load_frontier,
    save_frontier,
    update_frontier,
)

if TYPE_CHECKING:
    from pathlib import Path

    from autoresearch.pareto import Vector


def _entry(cid: str, vec: Vector, desc: str = "d") -> ParetoEntry:
    return ParetoEntry(
        candidate_id=cid, description=desc, change={"old_text": "o", "new_text": "n"}, vector=vec
    )


def test_dominates_strict() -> None:
    assert dominates((1, 1, 1, 1, 1), (0, 0, 0, 0, 0))
    assert dominates((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))


def test_dominates_equal_not_dominating() -> None:
    assert not dominates((1, 1, 1, 1, 1), (1, 1, 1, 1, 1))


def test_dominates_incomparable() -> None:
    assert not dominates((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    assert not dominates((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))


def test_update_frontier_drops_dominated() -> None:
    frontier = update_frontier(
        [_entry("c1", (0.1, 0.1, 0, 0, 0))], _entry("c2", (0.5, 0.5, 0, 0, 0))
    )
    assert [e.candidate_id for e in frontier] == ["c2"]


def test_update_frontier_rejects_dominated_candidate() -> None:
    frontier = update_frontier(
        [_entry("c1", (0.5, 0.5, 0, 0, 0))], _entry("c2", (0.1, 0.1, 0, 0, 0))
    )
    assert [e.candidate_id for e in frontier] == ["c1"]


def test_update_frontier_keeps_incomparable() -> None:
    frontier = update_frontier(
        [_entry("c1", (0.9, 0.1, 0, 0, 0))], _entry("c2", (0.1, 0.9, 0, 0, 0))
    )
    assert {e.candidate_id for e in frontier} == {"c1", "c2"}


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "frontier.json"
    save_frontier([_entry("c1", (0.5, 0.4, 0.3, 0.2, 0.1), desc="hi")], path)
    loaded = load_frontier(path)
    assert len(loaded) == 1
    assert loaded[0].candidate_id == "c1"
    assert loaded[0].vector == (0.5, 0.4, 0.3, 0.2, 0.1)
    assert loaded[0].change == {"old_text": "o", "new_text": "n"}


def test_load_frontier_absent_returns_empty(tmp_path: Path) -> None:
    assert load_frontier(tmp_path / "nope.json") == []
