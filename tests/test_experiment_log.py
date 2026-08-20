"""Unit tests for experiment_log.py — the ledger writer.

The row is a positional list that must line up with `HEADER` index for index.
Appending a column touches 2 places, and nothing caught a mismatch before.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest
from autoresearch import experiment_log
from autoresearch.experiment_log import HEADER, log_experiment
from autoresearch.metrics import compute_score

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "results.tsv"
    monkeypatch.setattr(experiment_log, "RESULTS_FILE", path)
    return path


def _log_one(**metrics: object) -> None:
    log_experiment(
        experiment_id="e1",
        loop="1",
        change_description="test",
        score=compute_score(metrics),
        accepted=True,
        git_sha="abc1234",
    )


def _read_one(ledger: Path, **metrics: object) -> dict[str, str]:
    """Log 1 row and read it back by column name."""
    _log_one(**metrics)
    with ledger.open(newline="") as handle:
        return next(iter(csv.DictReader(handle, delimiter="\t")))


def test_the_row_has_one_cell_per_column(ledger: Path) -> None:
    """A short or long row silently shifts every later column."""
    _log_one()
    data_row = ledger.read_text().splitlines()[1]
    assert len(data_row.split("\t")) == len(HEADER)


def test_act_latency_lands_under_its_own_name(ledger: Path) -> None:
    """The check that catches an index mismatch: a value written, then named."""
    assert _read_one(ledger, act_latency_p95_ms=187.0)["act_latency_p95_ms"] == "187"


def test_perceive_latency_lands_under_its_own_name(ledger: Path) -> None:
    assert _read_one(ledger, perceive_latency_p50_ms=1400.0)["perceive_latency_p50_ms"] == "1400"


def test_loop_arch_is_carried_through(ledger: Path) -> None:
    """Rows of different architectures are not comparable on turn latency."""
    assert _read_one(ledger, loop_arch="clocks")["loop_arch"] == "clocks"


def test_a_missing_metric_writes_an_empty_cell(ledger: Path) -> None:
    """The scenario and synth paths record no latency at all."""
    assert _read_one(ledger)["loop_arch"] == ""
