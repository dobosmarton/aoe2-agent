"""Experiment gate — refuse to merge a change without a recorded game row.

Every behavior-affecting change must have at least one full game recorded in
experiments/results.tsv before it merges (IMPROVEMENT-PLAN.md P0.1): without a
baseline row, regressions like the exp_0011 age hallucination or the exp_0013
town-bell collapse are only caught by luck.

Usage:
    python -m autoresearch.experiment_gate            # require a row at HEAD
    python -m autoresearch.experiment_gate --sha abc123
    python -m autoresearch.experiment_gate --any      # require any row at all

Exit code 0 when the required row exists, 1 otherwise (usable from CI or a
merge checklist). Run games with `just experiment "<description>"` on the VM
to produce rows.
"""

import argparse
import csv
import sys

from .experiment_log import RESULTS_FILE, get_git_sha


def _data_rows() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def check(sha: str | None = None, allow_any: bool = False) -> tuple[bool, str]:
    """Whether the ledger satisfies the gate; returns (ok, human message)."""
    rows = _data_rows()
    if allow_any:
        if rows:
            return True, f"ledger has {len(rows)} recorded game(s)"
        return False, f"ledger is empty: {RESULTS_FILE}"

    target = sha or get_git_sha()
    matched = [r for r in rows if r.get("git_sha") == target]
    if matched:
        scores = ", ".join(r.get("composite_score", "?") for r in matched)
        return True, f"{len(matched)} game(s) recorded at {target} (composite: {scores})"
    tail = rows[-3:]
    recent = (
        "; ".join(
            f"{r.get('experiment_id')}@{r.get('git_sha')} composite={r.get('composite_score')}"
            for r in tail
        )
        or "none"
    )
    return False, (
        f"no experiment row for {target} in {RESULTS_FILE}\n"
        f"most recent rows: {recent}\n"
        'record one on the VM with: just experiment "<what changed>"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate merges on a recorded experiment row")
    parser.add_argument("--sha", default=None, help="Git short SHA to require (default: HEAD)")
    parser.add_argument(
        "--any",
        action="store_true",
        dest="allow_any",
        help="Pass if the ledger has any data row at all (bootstrap mode)",
    )
    args = parser.parse_args()

    ok, message = check(sha=args.sha, allow_any=args.allow_any)
    print(("PASS: " if ok else "FAIL: ") + message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
