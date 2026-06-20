"""Promote the eval harness's best-F1 per-class thresholds into thresholds.py.

``evaluate_real.py --conf-sweep`` writes the best-F1 confidence threshold for
each class to ``recommended_thresholds`` inside eval_real_summary.json. This tool
turns those recommendations into the ``CLASS_THRESHOLDS`` mapping the detector and
server actually use, so wiring tuned thresholds after a retrain is one reviewable
command instead of hand-copying numbers.

Recommendations are *overlaid* on the current thresholds: classes the sweep
covered get their tuned value, everything else keeps its existing setting.

Usage:
    # Print the merged mapping for review (default — does not touch any file)
    python -m detection.inference.sync_thresholds path/to/eval_real_summary.json

    # Rewrite the CLASS_THRESHOLDS block in thresholds.py in place
    python -m detection.inference.sync_thresholds path/to/eval_real_summary.json --write

    # Use the synthetic split's recommendations instead of the real one
    python -m detection.inference.sync_thresholds summary.json --split synth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from .thresholds import CLASS_THRESHOLDS

if TYPE_CHECKING:
    from collections.abc import Mapping

Split = Literal["real", "synth"]

_BEGIN_MARKER = "# BEGIN GENERATED CLASS_THRESHOLDS"
_END_MARKER = "# END GENERATED CLASS_THRESHOLDS"
_THRESHOLDS_PATH = Path(__file__).with_name("thresholds.py")


def extract_recommendations(summary: object, split: Split) -> dict[str, float]:
    """Pull best-F1 per-class thresholds from a parsed eval_real_summary.json.

    Args:
        summary: The object returned by ``json.loads`` on the summary file.
        split: Which group's recommendations to read ("real" or "synth").

    Raises:
        ValueError: If the JSON shape is not a summary with the requested group's
            ``recommended_thresholds`` (e.g. the eval was run without --conf-sweep).
    """
    if not isinstance(summary, dict):
        raise ValueError("summary JSON is not an object")
    group = summary.get(split)
    if not isinstance(group, dict):
        raise ValueError(f"summary has no '{split}' group")
    recommended = group.get("recommended_thresholds")
    if not isinstance(recommended, dict):
        raise ValueError(
            f"'{split}' group has no recommended_thresholds — "
            "re-run evaluate_real.py with --conf-sweep"
        )
    out: dict[str, float] = {}
    for name, rec in recommended.items():
        if isinstance(rec, dict):
            threshold = rec.get("threshold")
            if isinstance(threshold, (int, float)):
                out[str(name)] = round(float(threshold), 2)
    return out


def build_class_thresholds(recommended: Mapping[str, float]) -> dict[str, float]:
    """Overlay swept recommendations onto the current hand-set thresholds."""
    return {**CLASS_THRESHOLDS, **recommended}


def render_block(thresholds: Mapping[str, float]) -> str:
    """Render the marked CLASS_THRESHOLDS source block (deterministic, name-sorted)."""
    lines = [
        _BEGIN_MARKER + "  (managed by detection.inference.sync_thresholds)",
        "CLASS_THRESHOLDS: dict[str, float] = {",
        *(f'    "{name}": {thresholds[name]:.2f},' for name in sorted(thresholds)),
        "}",
        _END_MARKER,
    ]
    return "\n".join(lines)


def rewrite_source(source: str, block: str) -> str:
    """Replace the marked CLASS_THRESHOLDS block in thresholds.py source text."""
    pattern = re.compile(re.escape(_BEGIN_MARKER) + r".*?" + re.escape(_END_MARKER), re.DOTALL)
    if not pattern.search(source):
        raise ValueError(f"markers {_BEGIN_MARKER!r} / {_END_MARKER!r} not found in thresholds.py")
    return pattern.sub(lambda _match: block, source)


class _SyncThresholdsArgs(argparse.Namespace):
    summary: str
    split: Split
    write: bool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote eval best-F1 per-class thresholds into thresholds.py"
    )
    parser.add_argument("summary", help="Path to eval_real_summary.json")
    parser.add_argument(
        "--split",
        choices=("real", "synth"),
        default="real",
        help="Which group's recommendations to use (default: real)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite thresholds.py in place (default: print the block only)",
    )
    args = parser.parse_args(namespace=_SyncThresholdsArgs())

    # json.loads is typed Any; narrow to object so extract_recommendations must
    # isinstance-check the shape rather than trusting it.
    summary = cast("object", json.loads(Path(args.summary).read_text()))
    recommended = extract_recommendations(summary, args.split)
    if not recommended:
        print("No recommendations found in the summary; nothing to do.")
        return 1

    merged = build_class_thresholds(recommended)
    block = render_block(merged)
    print(block)

    if args.write:
        source = _THRESHOLDS_PATH.read_text()
        _THRESHOLDS_PATH.write_text(rewrite_source(source, block))
        print(f"\nWrote {len(merged)} thresholds ({len(recommended)} tuned) to {_THRESHOLDS_PATH}")
    else:
        print("\n[dry-run] re-run with --write to update thresholds.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
