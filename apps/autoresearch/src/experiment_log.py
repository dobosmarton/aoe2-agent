"""Experiment ledger — TSV logging and git integration for autoresearch.

Maintains experiments/results.tsv with one row per experiment.
"""

import csv
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import structlog

from .metrics import GameScore

log = structlog.stdlib.get_logger()

# The ledger lives at the REPO ROOT (experiments/results.tsv) — the committed,
# reviewable record every change is measured against (IMPROVEMENT-PLAN.md P0.1).
# parents[3]: src -> autoresearch -> apps -> repo root.
RESULTS_FILE = Path(__file__).parents[3] / "experiments" / "results.tsv"

HEADER = [
    "experiment_id",
    "timestamp",
    "loop",
    "change_description",
    "composite_score",
    "survival",
    # `population` left the score in v2 (plan 2.3). The column stays because the
    # v1 rows hold real values in it; v2 rows write it empty.
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
    # Tournament (successive-halving) attribution; empty for single-candidate runs.
    "tournament_id",
    "candidate_id",
    "round",
    "game_in_round",
    # Executor health (T-533): fraction of turns where every LLM path failed.
    # Near 1.0 = the game was played by the reactive tier alone, so the row's
    # score reflects the fallback policy, not the executor under test (run 12).
    "llm_error_rate",
    # Turn latency (plan 0.3), and which phase led, as "ocr=11040 detect=890".
    "turn_latency_p50_ms",
    "turn_latency_p90_ms",
    "phase_latency_p50_ms",
    # Rows of different score versions are NOT comparable (plan 2.3).
    "score_version",
    # How fast the agent left the Dark Age. Empty on v1 rows, which never
    # recorded an age-up time.
    "age_speed",
    "feudal_time_s",
    # Phase 3 latency. The turn_latency_* columns above only compare within
    # one `loop_arch`: "turn" times a coupled tick, "clocks" the act loop.
    "act_latency_p95_ms",
    "perceive_latency_p50_ms",
    "loop_arch",
]

# 2 = age-weighted (plan 2.3). Version 1 weighted survival 0.30 and population
# 0.25 against age 0.20, which ranked the only Feudal game 3rd of 14.
SCORE_VERSION = 2


def _metric(score: GameScore, key: str, default: float = 0.0) -> float:
    """A numeric metric, or `default` when absent or non-numeric."""
    value = score.raw_metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _memories_used_count(metrics: Mapping[str, object]) -> int:
    """Total `[applied: …]` tags across every loaded memory."""
    used = metrics.get("memories_used")
    if not isinstance(used, dict):
        return 0
    counts: Iterable[object] = used.values()
    return sum(v for v in counts if isinstance(v, int))


def _format_seconds(value: object) -> str:
    """Whole seconds, or empty when the age was never reached."""
    return f"{value:.0f}" if isinstance(value, (int, float)) else ""


def _format_phase_latency(phases: object) -> str:
    """Render the per-phase p50 map as one cell: "ocr=11040 detect=890"."""
    if not isinstance(phases, dict) or not phases:
        return ""
    return " ".join(f"{name}={float(value):.0f}" for name, value in sorted(phases.items()))


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
    *,
    tournament_id: str = "",
    candidate_id: str = "",
    round_num: str = "",
    game_in_round: str = "",
) -> None:
    """Append an experiment result to the TSV ledger.

    The keyword-only tournament fields default to empty so single-candidate
    callers are unchanged; they attribute a row to a successive-halving run.
    """
    _ensure_results_file()

    if git_sha is None:
        git_sha = get_git_sha()

    memories_loaded = score.raw_metrics.get("memories_loaded")
    loaded_count = len(memories_loaded) if isinstance(memories_loaded, list) else 0

    row = [
        experiment_id,
        datetime.now(UTC).isoformat(timespec="seconds"),
        loop,
        change_description,
        f"{score.composite:.4f}",
        f"{score.survival:.4f}",
        "",  # population — scored in v1 only
        f"{score.age:.4f}",
        f"{score.economy:.4f}",
        f"{score.action_success:.4f}",
        str(score.raw_metrics.get("game_end_reason", "")),
        f"{_metric(score, 'turn_count'):.0f}",
        "true" if accepted else "false",
        git_sha if accepted else "(reverted)",
        str(loaded_count),
        str(_memories_used_count(score.raw_metrics)),
        tournament_id,
        candidate_id,
        round_num,
        game_in_round,
        f"{_metric(score, 'llm_error_rate'):.4f}",
        f"{_metric(score, 'turn_latency_p50_ms'):.0f}",
        f"{_metric(score, 'turn_latency_p90_ms'):.0f}",
        _format_phase_latency(score.raw_metrics.get("phase_latency_p50_ms")),
        str(SCORE_VERSION),
        f"{score.age_speed:.4f}",
        _format_seconds(score.raw_metrics.get("feudal_time_s")),
        f"{_metric(score, 'act_latency_p95_ms'):.0f}",
        f"{_metric(score, 'perceive_latency_p50_ms'):.0f}",
        str(score.raw_metrics.get("loop_arch", "")),
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
