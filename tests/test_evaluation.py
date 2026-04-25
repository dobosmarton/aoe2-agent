"""Pytest suite for the evaluation framework.

Three layers:
  1. Assertion DSL unit tests (no LLM, fast)
  2. Fixture YAML lint (no LLM, fast)
  3. Live scenario runs (requires ANTHROPIC_API_KEY; gated by --runlive)

Run modes:
    pytest tests/test_evaluation.py                # layers 1 & 2 only
    pytest tests/test_evaluation.py --runlive      # all three (costs ~$0.50)

The --runlive flag and `live` marker are configured in tests/conftest.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Layer 1: Assertion DSL unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def actions():
    return [
        {"type": "press", "key": "h", "rescan": True, "intent": "Go to TC"},
        {"type": "press", "key": "z", "intent": "Research Feudal"},
        {"type": "send_villager", "target_class": "tree", "intent": "sweep"},
    ]


def test_must_include_pass(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"must_include": {"type": "press", "key": "z"}}, actions=actions, reasoning="")
    assert failures == []


def test_must_include_fail(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"must_include": {"type": "build"}}, actions=actions, reasoning="")
    assert any("must_include FAILED" in f for f in failures)


def test_must_include_first_pass(actions):
    from evaluation.assertions import evaluate
    expected = {"must_include_first": [{"type": "press", "key": "h"}, {"type": "press", "key": "z"}]}
    failures = evaluate(expected, actions=actions, reasoning="")
    assert failures == []


def test_must_include_first_wrong_order(actions):
    from evaluation.assertions import evaluate
    expected = {"must_include_first": [{"type": "press", "key": "z"}]}
    failures = evaluate(expected, actions=actions, reasoning="")
    assert any("must_include_first FAILED at index 0" in f for f in failures)


def test_must_not_include_list(actions):
    from evaluation.assertions import evaluate
    expected = {"must_not_include": [{"type": "press", "key": "b"}, {"type": "queue_villager"}]}
    failures = evaluate(expected, actions=actions, reasoning="")
    assert failures == []


def test_must_not_include_hit(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"must_not_include": {"type": "press", "key": "h"}},
                        actions=actions, reasoning="")
    assert any("must_not_include FAILED" in f for f in failures)


def test_count_at_most_zero(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"count_at_most": {"type": "build", "n": 0}},
                        actions=actions, reasoning="")
    assert failures == []


def test_count_at_most_exceeded(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"count_at_most": {"type": "press", "n": 1}},
                        actions=actions, reasoning="")
    assert any("count_at_most FAILED" in f for f in failures)


def test_count_at_least_match():
    from evaluation.assertions import evaluate
    actions = [{"type": "build", "building_key": "a"}, {"type": "build", "building_key": "a"}]
    failures = evaluate({"count_at_least": {"type": "build", "building_key": "a", "n": 2}},
                        actions=actions, reasoning="")
    assert failures == []


def test_applied_memories_exact(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"applied_memories": ["foo"]}, actions=actions,
                        reasoning="[applied: foo] reasoning text")
    assert failures == []


def test_applied_memories_subset_extras_ok(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"applied_memories_subset": ["foo"]}, actions=actions,
                        reasoning="[applied: foo, bar] reasoning text")
    assert failures == []


def test_reasoning_contains_case_insensitive(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"reasoning_contains": "FEUDAL"}, actions=actions,
                        reasoning="going to feudal age")
    assert failures == []


def test_unknown_assertion_caught(actions):
    from evaluation.assertions import evaluate
    failures = evaluate({"bogus_assertion": True}, actions=actions, reasoning="")
    assert any("unknown assertion key" in f for f in failures)


# ---------------------------------------------------------------------------
# Layer 2: Fixture YAML lint
# ---------------------------------------------------------------------------


def _all_fixtures() -> list[Path]:
    scenarios_dir = REPO / "evaluation" / "scenarios"
    return sorted(scenarios_dir.glob("*.yaml"))


@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.stem)
def test_fixture_lint(fixture_path):
    """Each YAML fixture must parse and have the required schema fields."""
    import yaml
    data = yaml.safe_load(fixture_path.read_text())
    assert isinstance(data, dict), f"{fixture_path.name}: top-level must be a dict"
    assert "name" in data, f"{fixture_path.name}: missing 'name'"
    assert "inputs" in data, f"{fixture_path.name}: missing 'inputs'"
    assert "expected" in data, f"{fixture_path.name}: missing 'expected'"

    inputs = data["inputs"]
    for required in ("age", "resources", "detected_entities"):
        assert required in inputs, f"{fixture_path.name}: inputs missing {required!r}"

    # Each entity must have class + x + y
    for e in inputs["detected_entities"]:
        assert "class" in e and "x" in e and "y" in e, (
            f"{fixture_path.name}: entity missing required fields: {e}"
        )


# ---------------------------------------------------------------------------
# Layer 3: Live scenario runs (opt-in via --runlive)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.stem)
def test_scenario_runs(fixture_path):
    """Run a real scenario through ClaudeProvider. Requires ANTHROPIC_API_KEY."""
    from evaluation.runner import _load_dotenv, run_scenario
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = run_scenario(fixture_path)

    if result.skipped:
        pytest.skip(result.skip_reason)

    if not result.passed:
        pytest.fail(
            f"\n{fixture_path.name} failed:\n  "
            + "\n  ".join(result.failures)
            + f"\nReasoning: {result.reasoning[:300]}"
        )
