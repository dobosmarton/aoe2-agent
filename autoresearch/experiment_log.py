"""Experiment ledger — TSV logging and git integration for autoresearch.

Maintains experiments/results.tsv with one row per experiment.
"""

import csv
import subprocess
from datetime import UTC, datetime
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
    # New columns are appended (not inserted) so existing TSV rows on the VM
    # remain aligned when read with csv.DictReader — missing values become None.
    "memories_loaded",
    "memories_used_count",
]


def _ensure_results_file() -> None:
    """Create results file with header if missing, or upgrade a stale header.

    When new columns are appended to HEADER, old rows still have the original
    column count. csv.DictReader handles that gracefully (missing trailing
    fields read as None). The header line itself, however, must reflect the
    current HEADER so subsequent writes line up — otherwise new rows would
    have more fields than the header advertises, producing a "rest" key in
    DictReader output.
    """
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        with RESULTS_FILE.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(HEADER)
        return

    # File exists — check whether the header needs upgrading.
    with RESULTS_FILE.open(newline="") as f:
        existing_lines = f.readlines()
    if not existing_lines:
        with RESULTS_FILE.open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(HEADER)
        return

    current_header = existing_lines[0].rstrip("\n").split("\t")
    if current_header == HEADER:
        return

    # Header is stale — rewrite it, leaving data rows unchanged.
    log.info("results_tsv_header_upgraded", old_cols=len(current_header), new_cols=len(HEADER))
    with RESULTS_FILE.open("w", newline="") as f:
        f.write("\t".join(HEADER) + "\n")
        f.writelines(existing_lines[1:])


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
    with RESULTS_FILE.open() as f:
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

    memories_used = score.raw_metrics.get("memories_used", {}) or {}
    memories_loaded = score.raw_metrics.get("memories_loaded", []) or []
    memories_used_count = sum(int(v) for v in memories_used.values())

    row = [
        experiment_id,
        datetime.now(UTC).isoformat(timespec="seconds"),
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
        str(len(memories_loaded)),
        str(memories_used_count),
    ]

    with RESULTS_FILE.open("a", newline="") as f:
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
    with RESULTS_FILE.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))

    return rows[-n:]


def get_best_score(loop: str | None = None) -> float:
    """Get the best composite score from accepted experiments."""
    _ensure_results_file()

    best = 0.0
    with RESULTS_FILE.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("accepted") == "true" and (loop is None or row.get("loop") == loop):
                score = float(row.get("composite_score", 0))
                best = max(best, score)
    return best
