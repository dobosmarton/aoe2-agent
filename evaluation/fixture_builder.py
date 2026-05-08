"""Convert a saved game.txt log + screenshots/ into a YAML scenario fixture.

For a given turn N from a real game, this CLI scrapes:
  - resources/age from the LATEST strategist_response in the log (the
    strategist runs every ~10 turns, so this is approximate)
  - detected entity classes from ownership_classified lines within the turn

Outputs a fixture under evaluation/scenarios/regression/<name>.yaml. Goals,
real entity coordinates, and assertions still need hand-editing — this CLI
is a starting-point template, not a full export.

Usage:
    python -m evaluation.fixture_builder logs/2026_04_25/game.txt --turn 14
    python -m evaluation.fixture_builder logs/2026_04_25/game.txt --turn 14 \
        --out evaluation/scenarios/regression/exp_0013_turn_14.yaml \
        --name regression_exp_0013_turn_14_housing_stall
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_AGE = "Dark Age"
DEFAULT_POPULATION = "0/0"
PLACEHOLDER_BASE_X = 1500
PLACEHOLDER_BASE_Y = 800
PLACEHOLDER_X_STEP_PER_CLASS = 80
PLACEHOLDER_X_STEP_PER_INSTANCE = 20
PLACEHOLDER_Y_STEP_PER_CLASS = 60
PLACEHOLDER_CONFIDENCE = 0.85

_TURN_RE = re.compile(r"\[info\s*\]\s*iteration_start\s+iteration=(\d+)")
_RESOURCES_RE = re.compile(r"resources=(\{[^}]+\})")


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------


def _parse_kv_line(line: str) -> dict[str, str]:
    """Parse 'key1=val1 key2='val 2' key3=true' into a dict.

    Handles quoted values and bool/int conversion at the caller level.
    """
    result: dict[str, str] = {}
    for match in re.finditer(r"(\w+)=('([^']*)'|\"([^\"]*)\"|(\S+))", line):
        key = match.group(1)
        # First non-None of the three value-capture groups
        value = match.group(3) or match.group(4) or match.group(5) or ""
        result[key] = value
    return result


def _coerce_scalar(value: str) -> Any:
    """Convert a stringified scalar to int/float when it looks numeric, else keep as str."""
    if value is None:
        return value
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _parse_resources_dict(text: str) -> dict[str, Any]:
    """Parse a Python-style dict like '{food: 200, age: \"Dark Age\"}'.

    The log uses repr-style output — we use a permissive regex parser to
    avoid eval().
    """
    parsed: dict[str, Any] = {}
    pattern = re.compile(r"['\"]?(\w+)['\"]?\s*:\s*('([^']*)'|\"([^\"]*)\"|([\d/\.]+))")
    for match in pattern.finditer(text):
        key = match.group(1)
        value = match.group(3) or match.group(4) or match.group(5)
        parsed[key] = _coerce_scalar(value)
    return parsed


def _latest_strategist_resources(lines: list[str]) -> dict[str, Any]:
    """Return the resources dict from the LATEST strategist_response line.

    The strategist runs roughly every 10 turns, so for any specific target turn
    the nearest strategist reading may be several turns old. We pick the most
    recent one for simplicity — fixture authors should hand-correct if needed.
    """
    latest: dict[str, Any] = {}
    for line in lines:
        if "strategist_response" not in line:
            continue
        match = _RESOURCES_RE.search(line)
        if not match:
            continue
        candidate = _parse_resources_dict(match.group(1))
        if candidate:
            latest = candidate
    return latest


def _entity_classes_in_turn(lines: list[str], target_turn: int) -> dict[str, int]:
    """Count ownership_classified entity classes that appear during target_turn."""
    counts: dict[str, int] = defaultdict(int)
    in_target_turn = False
    for line in lines:
        turn_match = _TURN_RE.search(line)
        if turn_match:
            in_target_turn = int(turn_match.group(1)) == target_turn
        if in_target_turn and "ownership_classified" in line:
            class_name = _parse_kv_line(line).get("cls")
            if class_name:
                counts[class_name] += 1
    return counts


def _placeholder_entity(class_name: str, class_index: int, instance_index: int) -> dict:
    """Construct a placeholder entity with synthetic coordinates.

    Real coords aren't in the log — fixture authors should hand-edit them
    when precise placement matters.
    """
    return {
        "class": class_name,
        "x": PLACEHOLDER_BASE_X
        + class_index * PLACEHOLDER_X_STEP_PER_CLASS
        + instance_index * PLACEHOLDER_X_STEP_PER_INSTANCE,
        "y": PLACEHOLDER_BASE_Y + class_index * PLACEHOLDER_Y_STEP_PER_CLASS,
        "confidence": PLACEHOLDER_CONFIDENCE,
    }


def _entities_for_turn(lines: list[str], target_turn: int) -> list[dict]:
    """Build a synthetic entity list for `target_turn` from the log."""
    counts = _entity_classes_in_turn(lines, target_turn)
    entities: list[dict] = []
    for class_index, (class_name, count) in enumerate(sorted(counts.items())):
        for instance_index in range(count):
            entities.append(_placeholder_entity(class_name, class_index, instance_index))
    return entities


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------


def _description(log_path: Path, turn: int) -> str:
    return (
        f"Auto-generated from {log_path}, turn {turn}. "
        f"Hand-edit `expected:` and possibly `detected_entities` coords "
        f"before using. Goals are not currently scraped from the log "
        f"(strategist_goals_updated only logs counts) — populate manually."
    )


def _resources_block(scraped: dict[str, Any]) -> dict[str, Any]:
    return {
        "food": scraped.get("food", 0),
        "wood": scraped.get("wood", 0),
        "gold": scraped.get("gold", 0),
        "stone": scraped.get("stone", 0),
        "population": scraped.get("population", DEFAULT_POPULATION),
    }


def _build_fixture(log_path: Path, turn: int, name: str) -> dict:
    lines = log_path.read_text().splitlines()
    scraped = _latest_strategist_resources(lines)
    entities = _entities_for_turn(lines, turn)

    return {
        "name": name,
        "description": _description(log_path, turn),
        "inputs": {
            "age": scraped.get("age", DEFAULT_AGE),
            "resources": _resources_block(scraped),
            "detected_entities": entities,
            "goals": [],
            "memories": [],
            "recent_turns": [],
        },
        "screenshot": None,
        "expected": {
            # PLACEHOLDER — write the actual assertion(s) before running.
            "must_not_include": [
                {"type": "press", "key": "b"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_fixture_name(log_path: Path, turn: int) -> str:
    return f"regression_{log_path.parent.name}_turn_{turn:02d}"


def _default_output_path(name: str) -> Path:
    return REPO / "evaluation" / "scenarios" / "regression" / f"{name}.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YAML scenario fixture from a real game log."
    )
    parser.add_argument("log", type=Path, help="Path to logs/<date>/game.txt")
    parser.add_argument("--turn", "-t", type=int, required=True, help="Iteration number")
    parser.add_argument("--name", help="Fixture name (default: derived from log + turn)")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Output YAML path (default: scenarios/regression/<name>.yaml)",
    )
    return parser.parse_args()


def _print_summary(out: Path, fixture: dict) -> None:
    print(f"Wrote fixture: {out}")
    print(f"  age:       {fixture['inputs']['age']}")
    print(f"  resources: {fixture['inputs']['resources']}")
    print(f"  entities:  {len(fixture['inputs']['detected_entities'])} (placeholder coords)")
    print()
    print("NEXT: hand-edit expected/goals/coords before running:")
    print(f"  python -m evaluation.runner {out}")


def main() -> int:
    args = _parse_args()
    if not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        return 1

    name = args.name or _default_fixture_name(args.log, args.turn)
    out = args.out or _default_output_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)

    fixture = _build_fixture(args.log, args.turn, name)

    try:
        import yaml

        out.write_text(yaml.dump(fixture, sort_keys=False, default_flow_style=False))
    except ImportError:
        print("PyYAML not installed — cannot write fixture", file=sys.stderr)
        return 1

    _print_summary(out, fixture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
