"""Unit tests for evaluation/log_to_scenario.py.

Synthetic structlog tapes exercise the parser, interesting-turn detector,
and YAML emitter. Tests are offline (no API key, no real game logs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from evaluation.log_to_scenario import (
    TurnSnapshot,
    emit_fixture,
    find_age_transitions,
    parse_log,
)

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_LOG = """\
2026-04-25 10:37:06 [info     ] iteration_start                iteration=1
2026-04-25 10:37:10 [debug    ] detected_entities_set          count=12
2026-04-25 10:37:24 [info     ] strategist_response            goal_count=5 reasoning='reading bar' resources={'food': 200, 'wood': 200, 'gold': 100, 'stone': 200, 'population': '4/5', 'age': 'Dark Age'}
2026-04-25 10:37:57 [debug    ] claude_response                age='Dark Age' reasoning='Need a house at 4/5 pop'
2026-04-25 10:37:57 [info     ] iteration_start                iteration=2
2026-04-25 10:38:30 [debug    ] detected_entities_set          count=20
2026-04-25 10:38:54 [info     ] strategist_response            goal_count=4 reasoning='re-reading' resources={'food': 520, 'wood': 100, 'gold': 100, 'stone': 200, 'population': '22/25', 'age': 'Dark Age'}
2026-04-25 10:39:14 [debug    ] claude_response                age='Dark Age' reasoning='Time to advance to Feudal'
2026-04-25 10:39:14 [info     ] iteration_start                iteration=3
2026-04-25 10:39:50 [info     ] strategist_response            goal_count=3 resources={'food': 30, 'wood': 50, 'gold': 100, 'stone': 200, 'population': '22/25', 'age': 'Feudal Age'}
2026-04-25 10:40:00 [debug    ] claude_response                age='Feudal Age' reasoning='In Feudal now, build market'
"""


def _write_log(tmp_path: Path) -> Path:
    log = tmp_path / "game.txt"
    log.write_text(SAMPLE_LOG)
    return log


# ---------------------------------------------------------------------------
# parse_log
# ---------------------------------------------------------------------------


def test_parse_log_extracts_three_turns(tmp_path):
    log = _write_log(tmp_path)
    turns = parse_log(log)
    assert len(turns) == 3
    assert turns[0].iteration == 1
    assert turns[1].iteration == 2
    assert turns[2].iteration == 3


def test_parse_log_extracts_resources_dict(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    assert turns[0].resources == {
        "food": 200,
        "wood": 200,
        "gold": 100,
        "stone": 200,
        "population": "4/5",
        "age": "Dark Age",
    }


def test_parse_log_extracts_age_from_claude_response(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    assert turns[0].age == "Dark Age"
    assert turns[2].age == "Feudal Age"


def test_parse_log_captures_reasoning_preview(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    assert "Need a house" in turns[0].reasoning
    assert "Time to advance" in turns[1].reasoning


def test_parse_log_captures_entity_count(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    assert turns[0].entity_count == 12
    assert turns[1].entity_count == 20


def test_parse_log_captures_timestamp(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    assert turns[0].timestamp == "2026-04-25 10:37:06"


def test_parse_log_handles_malformed_lines(tmp_path):
    """Garbage lines and partial lines must not crash the parser."""
    log = tmp_path / "game.txt"
    log.write_text(
        "PS C:\\> python -m foo\n"
        "Some random output\n"
        "2026-04-25 10:37:06 [info     ] iteration_start                iteration=1\n"
        "WARNING: malformed\n"
        "2026-04-25 10:37:24 [info     ] strategist_response            goal_count=5 resources={'food': 100}\n"
    )
    turns = parse_log(log)
    assert len(turns) == 1
    assert turns[0].resources == {"food": 100}


def test_parse_log_skips_malformed_resources_dict(tmp_path):
    """A resources= dict that fails ast.literal_eval must not crash the run."""
    log = tmp_path / "game.txt"
    log.write_text(
        "2026-04-25 10:37:06 [info     ] iteration_start                iteration=1\n"
        "2026-04-25 10:37:24 [info     ] strategist_response            resources={broken}\n"
    )
    turns = parse_log(log)
    assert len(turns) == 1
    assert turns[0].resources is None  # parse failed cleanly


# ---------------------------------------------------------------------------
# find_age_transitions
# ---------------------------------------------------------------------------


def test_find_age_transitions_flags_transition(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    interesting = find_age_transitions(turns)
    assert len(interesting) == 1
    assert interesting[0].iteration == 3  # the Dark Age → Feudal Age transition
    assert interesting[0].age == "Feudal Age"


def test_find_age_transitions_empty_when_no_transitions():
    turns = [
        TurnSnapshot(iteration=1, timestamp="t1", age="Dark Age"),
        TurnSnapshot(iteration=2, timestamp="t2", age="Dark Age"),
    ]
    assert find_age_transitions(turns) == []


# ---------------------------------------------------------------------------
# Carry-forward behaviour (strategist runs every 3-10 turns, not every turn)
# ---------------------------------------------------------------------------


def test_resources_carry_forward_when_no_strategist_response(tmp_path):
    """Turns without a strategist_response inherit the last strategist reading.

    The strategist runs as a background async task every 3-10 turns, so most
    intermediate turns carry the same resource state as the last reading.
    """
    log = tmp_path / "game.txt"
    log.write_text(
        "2026-04-25 10:00:00 [info     ] iteration_start                iteration=1\n"
        "2026-04-25 10:00:01 [info     ] strategist_response            resources={'food': 200, 'wood': 100, 'gold': 0, 'stone': 0, 'population': '4/5', 'age': 'Dark Age'}\n"
        "2026-04-25 10:00:02 [debug    ] claude_response                age='Dark Age' reasoning='ok'\n"
        "2026-04-25 10:00:03 [info     ] iteration_start                iteration=2\n"
        "2026-04-25 10:00:04 [debug    ] claude_response                age='Dark Age' reasoning='no strategist this turn'\n"
        "2026-04-25 10:00:05 [info     ] iteration_start                iteration=3\n"
        "2026-04-25 10:00:06 [debug    ] claude_response                age='Dark Age' reasoning='still no strategist'\n"
    )
    turns = parse_log(log)
    assert len(turns) == 3
    # Turn 2 and 3 carry forward turn 1's strategist resources
    assert turns[1].resources == turns[0].resources
    assert turns[2].resources == turns[0].resources


def test_strategist_age_takes_precedence_over_claude_age(tmp_path):
    """The LLM has been observed to misreport age; strategist OCR is authoritative."""
    log = tmp_path / "game.txt"
    log.write_text(
        "2026-04-25 10:00:00 [info     ] iteration_start                iteration=1\n"
        "2026-04-25 10:00:01 [info     ] strategist_response            resources={'food': 200, 'age': 'Dark Age'}\n"
        "2026-04-25 10:00:02 [debug    ] claude_response                age='Feudal Age' reasoning='LLM hallucinating'\n"
    )
    turns = parse_log(log)
    assert turns[0].age == "Dark Age"  # strategist wins, not claude's "Feudal Age"


# ---------------------------------------------------------------------------
# emit_fixture
# ---------------------------------------------------------------------------


def test_emit_fixture_produces_valid_yaml(tmp_path):
    """The emitted YAML must parse and pass the same lint as hand-written fixtures."""
    turns = parse_log(_write_log(tmp_path))
    yaml_text = emit_fixture(turns[0], name="my_test_turn")
    data = yaml.safe_load(yaml_text)

    assert data["name"] == "my_test_turn"
    assert "inputs" in data
    assert data["inputs"]["age"] == "Dark Age"
    assert data["inputs"]["resources"]["food"] == 200
    assert data["inputs"]["resources"]["population"] == "4/5"

    # detected_entities is a placeholder list with the required schema fields
    entities = data["inputs"]["detected_entities"]
    assert isinstance(entities, list) and entities
    for e in entities:
        assert "class" in e and "x" in e and "y" in e


def test_emit_fixture_default_name_uses_iteration(tmp_path):
    turns = parse_log(_write_log(tmp_path))
    yaml_text = emit_fixture(turns[2])
    assert "turn_3_snapshot" in yaml_text


def test_emit_fixture_handles_special_chars_in_reasoning():
    """Reasoning containing quotes, colons, and newlines must not corrupt YAML."""
    turn = TurnSnapshot(
        iteration=7,
        timestamp="2026-04-25 10:00:00",
        age="Feudal Age",
        resources={"food": 200, "wood": 100, "gold": 0, "stone": 0, "population": "10/15"},
        reasoning='I should "queue villager": don\'t skip housing — pop_cap: 15.\nNext: build mill.',
    )
    yaml_text = emit_fixture(turn, name="quote_colon_test")
    data = yaml.safe_load(yaml_text)  # would fail loudly under f-string emission
    assert data["name"] == "quote_colon_test"
    assert data["inputs"]["resources"]["food"] == 200
    assert data["inputs"]["resources"]["population"] == "10/15"
