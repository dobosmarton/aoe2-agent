"""Experiment ledger — TSV logging and git integration for autoresearch.

Maintains experiments/results.tsv with one row per experiment.
"""

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import structlog

from .metrics import GameScore

log = structlog.get_logger()

RESULTS_FILE = Path(__file__).parent.parent / "experiments" / "results.tsv"

HEADER = [
    "experiment_id",
    "timestamp",
    "loop",
    "change_description",
    "composite_score",
    "survival",
    "population",
    "age",
    "economy",
    "action_success",
    "game_end_reason",
    "turn_count",
    "accepted",
    "git_sha",
]


def _ensure_results_file() -> None:
    """Create results file with header if it doesn't exist."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(HEADER)


def get_git_sha() -> str:
    """Get current short git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_next_experiment_id() -> str:
    """Generate next experiment ID based on existing entries."""
    _ensure_results_file()
    count = 0
    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for _ in reader:
            count += 1
    return f"exp_{count + 1:04d}"


def log_experiment(
    experiment_id: str,
    loop: str,
    change_description: str,
    score: GameScore,
    accepted: bool,
    git_sha: str | None = None,
) -> None:
    """Append an experiment result to the TSV ledger."""
    _ensure_results_file()

    if git_sha is None:
        git_sha = get_git_sha()

    row = [
        experiment_id,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        loop,
        change_description,
        f"{score.composite:.4f}",
        f"{score.survival:.4f}",
        f"{score.population:.4f}",
        f"{score.age:.4f}",
        f"{score.economy:.4f}",
        f"{score.action_success:.4f}",
        score.raw_metrics.get("game_end_reason", ""),
        str(score.raw_metrics.get("turn_count", 0)),
        "true" if accepted else "false",
        git_sha if accepted else "(reverted)",
    ]

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)

    log.info(
        "experiment_logged",
        experiment_id=experiment_id,
        loop=loop,
        score=score.composite,
        accepted=accepted,
    )


def get_recent_experiments(n: int = 5) -> list[dict]:
    """Read the last N experiments from the ledger."""
    _ensure_results_file()

    rows = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))

    return rows[-n:]


def get_best_score(loop: str | None = None) -> float:
    """Get the best composite score from accepted experiments."""
    _ensure_results_file()

    best = 0.0
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("accepted") == "true":
                if loop is None or row.get("loop") == loop:
                    score = float(row.get("composite_score", 0))
                    best = max(best, score)
    return best
