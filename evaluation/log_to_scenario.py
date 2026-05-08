"""Convert a real-game structlog stream into scenario YAML stubs.

A real game produces a structlog tape (`logs/YYYY_MM_DD/game.txt`) with
~30-60 iterations of strategist outputs, executor reasoning, and tool
actions. This tool extracts per-turn state snapshots and emits YAML
fixture stubs in the existing scenario schema. The user fills in
`detected_entities:` (not present in logs — only counts are) and
the `expected:` assertion block, then commits the regression fixture.

CLI:
    python -m evaluation.log_to_scenario logs/2026_04_25/game.txt --list
    python -m evaluation.log_to_scenario logs/2026_04_25/game.txt --turn 8 \
        --out evaluation/scenarios/regression/turn_8_food_emergency.yaml
    python -m evaluation.log_to_scenario logs/2026_04_25/game.txt --auto \
        --out-dir evaluation/scenarios/regression/

Auto-detect heuristic flags turns at age transitions — those are the
biggest behavioral discontinuities and the most valuable regression points.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"\[(?P<level>\w+)\s*\]\s+"
    r"(?P<event>\w+)\s*"
    r"(?P<rest>.*)$"
)

# Targeted regexes for the fields we care about. Each works on the `rest:`
# portion of a parsed log line — the key=value tail after the event name.
ITERATION_RE = re.compile(r"iteration=(\d+)")
GOAL_COUNT_RE = re.compile(r"goal_count=(\d+)")
AGE_RE = re.compile(r"age='([^']+)'")
REASONING_RE = re.compile(r"reasoning='((?:[^'\\]|\\.)*)'", re.DOTALL)
RESOURCES_RE = re.compile(r"resources=(\{[^}]+\})")

DEFAULT_PLACEHOLDER_ENTITIES = [
    {"class": "town_center", "x": 960, "y": 540},
    {"class": "villager", "x": 870, "y": 500},
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TurnSnapshot:
    iteration: int
    timestamp: str
    age: str | None = None
    resources: dict | None = None
    reasoning: str | None = None
    goal_count: int | None = None
    entity_count: int | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_log(log_path: Path) -> list[TurnSnapshot]:
    """Walk the log line-by-line, accumulating a TurnSnapshot per iteration.

    Each iteration_start opens a new turn. State that doesn't change every
    turn — strategist resources and the strategist's age reading — carries
    forward to subsequent turns until overridden, since the strategist
    runs every 3-10 turns rather than every turn. claude_response.age is
    used only as a fallback because the LLM has been observed to misreport
    age (mid-game oscillations between Dark/Feudal that are physically
    impossible), whereas the strategist's OCR is more reliable.
    """
    turns: list[TurnSnapshot] = []
    current: TurnSnapshot | None = None
    last_resources: dict | None = None
    last_strategist_age: str | None = None

    for raw_line in log_path.read_text().splitlines():
        match = LOG_LINE_RE.match(raw_line)
        if not match:
            continue
        event = match["event"]
        rest = match["rest"]
        ts = match["ts"]

        if event == "iteration_start":
            iter_match = ITERATION_RE.search(rest)
            if iter_match:
                if current is not None:
                    turns.append(current)
                current = TurnSnapshot(iteration=int(iter_match.group(1)), timestamp=ts)
                # Strategist runs less often than every turn — carry forward.
                if last_resources is not None:
                    current.resources = last_resources
                if last_strategist_age is not None:
                    current.age = last_strategist_age
            continue

        if current is None:
            continue

        if event == "strategist_response":
            res_match = RESOURCES_RE.search(rest)
            if res_match:
                try:
                    parsed = ast.literal_eval(res_match.group(1))
                    current.resources = parsed
                    last_resources = parsed
                    if "age" in parsed:
                        current.age = parsed["age"]
                        last_strategist_age = parsed["age"]
                except (ValueError, SyntaxError):
                    pass
            goal_match = GOAL_COUNT_RE.search(rest)
            if goal_match:
                current.goal_count = int(goal_match.group(1))

        elif event == "claude_response":
            # Strategist age is authoritative; only use claude's age if no
            # strategist reading is available for this turn.
            if current.age is None:
                age_match = AGE_RE.search(rest)
                if age_match:
                    current.age = age_match.group(1)
            reason_match = REASONING_RE.search(rest)
            if reason_match:
                current.reasoning = reason_match.group(1)

        elif event == "detected_entities_set":
            count_match = re.search(r"count=(\d+)", rest)
            if count_match:
                current.entity_count = int(count_match.group(1))

    if current is not None:
        turns.append(current)
    return turns


# ---------------------------------------------------------------------------
# Auto-detect interesting turns
# ---------------------------------------------------------------------------


def find_age_transitions(turns: list[TurnSnapshot]) -> list[TurnSnapshot]:
    """Flag turns where the age changed from the previous turn.

    Age transitions are the biggest behavioral discontinuity in AoE2 and
    the most valuable regression points to capture.
    """
    interesting: list[TurnSnapshot] = []
    last_age: str | None = None
    for turn in turns:
        if turn.age and last_age and turn.age != last_age:
            interesting.append(turn)
        if turn.age:
            last_age = turn.age
    return interesting


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

_EXPECTED_TEMPLATE = """
# expected:
#   must_include:
#     type: <action_type>
"""


def emit_fixture(turn: TurnSnapshot, *, name: str | None = None) -> str:
    """Render a TurnSnapshot as a single-turn fixture YAML stub.

    The user MUST fill in `detected_entities:` (logs only carry counts,
    not coordinates) and add an `expected:` block before the fixture is
    runnable. The reasoning preview is preserved in the description for
    context. YAML is serialized via `yaml.safe_dump` so reasoning containing
    quotes, colons, or newlines doesn't corrupt the output.
    """
    fixture_name = name or f"turn_{turn.iteration}_snapshot"
    resources = turn.resources or {}
    age = turn.age or resources.get("age", "Dark Age")
    reasoning_preview = (turn.reasoning or "").replace("\\n", " ")[:200]

    entity_note = (
        f"Real game had {turn.entity_count} entities at this turn."
        if turn.entity_count
        else "Entity coordinates are not in logs — fill these in by hand."
    )

    fixture_data = {
        "name": fixture_name,
        "description": (
            f"Auto-generated from log iteration {turn.iteration} ({turn.timestamp}). "
            f'Reasoning preview: "{reasoning_preview}...". '
            f"TODO: fill in detected_entities (logs carry only counts, not coords) "
            f"and add an `expected:` assertion block. "
            f"{entity_note}"
        ),
        "inputs": {
            "age": age,
            "resources": {
                "food": int(resources.get("food", 0)),
                "wood": int(resources.get("wood", 0)),
                "gold": int(resources.get("gold", 0)),
                "stone": int(resources.get("stone", 0)),
                "population": str(resources.get("population", "0/0")),
            },
            "detected_entities": [dict(e) for e in DEFAULT_PLACEHOLDER_ENTITIES],
        },
    }

    body = yaml.safe_dump(fixture_data, sort_keys=False, default_flow_style=False)
    return body + _EXPECTED_TEMPLATE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_list(turns: list[TurnSnapshot]) -> None:
    """Print a one-line summary per turn."""
    print(f"Found {len(turns)} turn(s):\n")
    for turn in turns:
        age = turn.age or "?"
        food = (turn.resources or {}).get("food", "?")
        pop = (turn.resources or {}).get("population", "?")
        print(
            f"  turn {turn.iteration:>3}  {turn.timestamp}  "
            f"age={age:<11}  food={food:<5}  pop={pop}"
        )


def _cmd_one_turn(turns: list[TurnSnapshot], turn_num: int, out: Path | None) -> int:
    snapshot = next((t for t in turns if t.iteration == turn_num), None)
    if snapshot is None:
        print(f"No turn {turn_num} in log.", file=sys.stderr)
        return 1
    yaml_text = emit_fixture(snapshot)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml_text)
        print(f"Wrote {out}")
    else:
        print(yaml_text)
    return 0


def _cmd_auto(turns: list[TurnSnapshot], out_dir: Path) -> int:
    interesting = find_age_transitions(turns)
    if not interesting:
        print("No age transitions detected.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for turn in interesting:
        slug = f"turn_{turn.iteration}_age_to_{(turn.age or 'unknown').replace(' ', '_').lower()}"
        path = out_dir / f"{slug}.yaml"
        path.write_text(emit_fixture(turn, name=slug))
        print(f"Wrote {path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a real-game structlog stream into scenario YAML stubs."
    )
    parser.add_argument("log", type=Path, help="Path to logs/<date>/game.txt")
    parser.add_argument("--list", action="store_true", help="Print one-line summary of every turn")
    parser.add_argument("--turn", type=int, help="Emit a single turn's fixture")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect interesting turns and emit one fixture per turn",
    )
    parser.add_argument(
        "--out", type=Path, help="Output path (single-turn mode); print to stdout if omitted"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory (auto mode)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        return 1

    turns = parse_log(args.log)
    if not turns:
        print("No turns parsed from log.", file=sys.stderr)
        return 1

    if args.list:
        _cmd_list(turns)
        return 0
    if args.turn is not None:
        return _cmd_one_turn(turns, args.turn, args.out)
    if args.auto:
        if not args.out_dir:
            print("--auto requires --out-dir", file=sys.stderr)
            return 1
        return _cmd_auto(turns, args.out_dir)

    print("Specify one of: --list, --turn N, --auto", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
