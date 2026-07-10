"""Tests for the experiment merge gate (IMPROVEMENT-PLAN.md P0.1)."""

from pathlib import Path

import pytest
from autoresearch import experiment_gate
from autoresearch.experiment_log import HEADER


def _write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["\t".join(HEADER)]
    for row in rows:
        lines.append("\t".join(row.get(col, "") for col in HEADER))
    path.write_text("\n".join(lines) + "\n")


def _row(sha: str, composite: str = "0.5000") -> dict[str, str]:
    return {
        "experiment_id": "exp_0001",
        "git_sha": sha,
        "composite_score": composite,
        "accepted": "true",
    }


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "results.tsv"
    monkeypatch.setattr(experiment_gate, "RESULTS_FILE", path)
    return path


def test_missing_ledger_fails(ledger: Path) -> None:
    ok, message = experiment_gate.check(sha="abc1234")
    assert not ok
    assert "no experiment row" in message


def test_empty_ledger_fails_even_with_any(ledger: Path) -> None:
    _write_ledger(ledger, [])
    assert not experiment_gate.check(allow_any=True)[0]
    assert not experiment_gate.check(sha="abc1234")[0]


def test_row_at_sha_passes(ledger: Path) -> None:
    _write_ledger(ledger, [_row("abc1234")])
    ok, message = experiment_gate.check(sha="abc1234")
    assert ok
    assert "abc1234" in message


def test_row_at_other_sha_fails_but_shows_recent(ledger: Path) -> None:
    _write_ledger(ledger, [_row("abc1234")])
    ok, message = experiment_gate.check(sha="fff0000")
    assert not ok
    assert "abc1234" in message  # recent rows are surfaced for context


def test_any_mode_passes_on_nonempty_ledger(ledger: Path) -> None:
    _write_ledger(ledger, [_row("abc1234")])
    ok, _ = experiment_gate.check(allow_any=True)
    assert ok
