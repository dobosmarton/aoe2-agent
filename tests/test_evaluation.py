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
# differs_from_baseline_by tests
# ---------------------------------------------------------------------------


def test_differs_must_include_pass():
    """Variant has the pattern, baseline doesn't → pass."""
    from evaluation.assertions import evaluate
    baseline = [{"type": "queue_villager"}]
    variant = [{"type": "queue_villager"}, {"type": "build", "building_key": "q"}]
    expected = {"differs_from_baseline_by": {"must_include": {"type": "build", "building_key": "q"}}}
    failures = evaluate(expected, actions=variant, reasoning="", baseline_actions=baseline)
    assert failures == []


def test_differs_must_include_fail_when_pattern_in_baseline():
    """Pattern in BOTH baseline and variant → fail (no actual difference)."""
    from evaluation.assertions import evaluate
    baseline = [{"type": "build", "building_key": "q"}]
    variant = [{"type": "build", "building_key": "q"}, {"type": "queue_villager"}]
    expected = {"differs_from_baseline_by": {"must_include": {"type": "build", "building_key": "q"}}}
    failures = evaluate(expected, actions=variant, reasoning="", baseline_actions=baseline)
    assert any("differs_from_baseline_by.must_include FAILED" in f for f in failures)


def test_differs_must_not_include_pass():
    """Baseline has pattern, variant doesn't → pass."""
    from evaluation.assertions import evaluate
    baseline = [{"type": "queue_villager"}, {"type": "press", "key": "q"}]
    variant = [{"type": "build", "building_key": "a"}]
    expected = {"differs_from_baseline_by": {"must_not_include": {"type": "queue_villager"}}}
    failures = evaluate(expected, actions=variant, reasoning="", baseline_actions=baseline)
    assert failures == []


def test_differs_no_baseline_fails_explicitly():
    """differs_from_baseline_by without a baseline must produce a clear error."""
    from evaluation.assertions import evaluate
    expected = {"differs_from_baseline_by": {"must_include": {"type": "build"}}}
    failures = evaluate(expected, actions=[], reasoning="", baseline_actions=None)
    assert any("no baseline available" in f for f in failures)


def test_differs_combined_must_include_and_must_not_include():
    """Both directions can be checked in one assertion."""
    from evaluation.assertions import evaluate
    baseline = [{"type": "queue_villager"}]
    variant = [{"type": "build", "building_key": "a"}]
    expected = {
        "differs_from_baseline_by": {
            "must_include": {"type": "build", "building_key": "a"},
            "must_not_include": {"type": "queue_villager"},
        }
    }
    failures = evaluate(expected, actions=variant, reasoning="", baseline_actions=baseline)
    assert failures == []


# ---------------------------------------------------------------------------
# Layer 1b: variants expansion (offline, no LLM)
# ---------------------------------------------------------------------------


def test_expand_variants_no_variants_returns_single_anonymous():
    from evaluation.runner import _expand_variants
    fixture = {"name": "x", "inputs": {"age": "Dark Age"}, "expected": {"x": 1}}
    expanded = _expand_variants(fixture)
    assert len(expanded) == 1
    assert expanded[0]["_variant_name"] is None
    assert expanded[0]["inputs"]["age"] == "Dark Age"


def test_expand_variants_overlays_memories_on_base_inputs():
    from evaluation.runner import _expand_variants
    base_mem = {"title": "base"}
    variant_mem = {"title": "variant"}
    fixture = {
        "name": "x",
        "inputs": {"age": "Dark Age", "memories": [base_mem]},
        "variants": [
            {"name": "no_mem", "memories": [], "expected": {"a": 1}},
            {"name": "swap_mem", "memories": [variant_mem], "expected": {"b": 2}},
        ],
    }
    expanded = _expand_variants(fixture)
    assert len(expanded) == 2
    assert expanded[0]["_variant_name"] == "no_mem"
    assert expanded[1]["_variant_name"] == "swap_mem"
    # Variants override `memories` only — other base inputs are preserved
    assert expanded[0]["inputs"]["age"] == "Dark Age"
    assert expanded[1]["inputs"]["age"] == "Dark Age"
    assert expanded[0]["inputs"]["memories"] == []
    assert expanded[1]["inputs"]["memories"] == [variant_mem]
    # Per-variant `expected:` is used, not the top-level one
    assert expanded[0]["expected"] == {"a": 1}
    assert expanded[1]["expected"] == {"b": 2}


def test_expand_variants_inherits_base_memories_when_variant_omits():
    from evaluation.runner import _expand_variants
    base_mem = {"title": "base"}
    fixture = {
        "name": "x",
        "inputs": {"age": "Dark Age", "memories": [base_mem]},
        "variants": [{"name": "v0", "expected": {}}],  # no memories key
    }
    expanded = _expand_variants(fixture)
    assert expanded[0]["inputs"]["memories"] == [base_mem]


def test_scenario_display_name_with_and_without_variant():
    from evaluation.runner import _scenario_display_name
    from pathlib import Path
    path = Path("evaluation/scenarios/x.yaml")
    assert _scenario_display_name(path, None) == "x"
    assert _scenario_display_name(path, "baseline") == "x [baseline]"


# ---------------------------------------------------------------------------
# Layer 1c: Memory dir isolation (offline, no LLM)
# ---------------------------------------------------------------------------


def test_isolate_memories_dir_normal_flow_backs_up_and_restores(tmp_path, monkeypatch):
    """Real memories are moved aside, fixtures planted, then restored on exit."""
    from autoresearch import memory_chain
    from evaluation.runner import _isolate_memories_dir

    fake_memories = tmp_path / "memories"
    fake_memories.mkdir()
    (fake_memories / "real.md").write_text("real content")
    monkeypatch.setattr(memory_chain, "MEMORIES_DIR", fake_memories)

    fixture = {"title": "fixture_rule", "content": "fixture content"}
    with _isolate_memories_dir([fixture]):
        assert not (fake_memories / "real.md").exists()
        planted = list(fake_memories.glob("*.md"))
        assert len(planted) == 1 and "fixture_rule" in planted[0].name

    assert (fake_memories / "real.md").read_text() == "real content"
    leftover = [p for p in fake_memories.glob("*.md") if "fixture_rule" in p.name]
    assert not leftover


def test_isolate_memories_dir_raises_on_orphan_backup(tmp_path, monkeypatch):
    """An orphan backup from a crashed prior run blocks new runs (no silent data loss)."""
    from autoresearch import memory_chain
    from evaluation.runner import _isolate_memories_dir

    fake_memories = tmp_path / "memories"
    fake_backup = tmp_path / "memories_eval_backup"
    fake_memories.mkdir()
    fake_backup.mkdir()
    (fake_backup / "real_user_memory.md").write_text("important user data")
    monkeypatch.setattr(memory_chain, "MEMORIES_DIR", fake_memories)

    with pytest.raises(RuntimeError, match="orphan eval backup"):
        with _isolate_memories_dir([]):
            pass

    # The orphan backup must remain untouched — that's the whole point.
    assert fake_backup.exists()
    assert (fake_backup / "real_user_memory.md").read_text() == "important user data"


def test_isolate_memories_dir_no_existing_memories(tmp_path, monkeypatch):
    """Fresh state (no memories dir) works without errors."""
    from autoresearch import memory_chain
    from evaluation.runner import _isolate_memories_dir

    fake_memories = tmp_path / "memories"  # does NOT exist yet
    monkeypatch.setattr(memory_chain, "MEMORIES_DIR", fake_memories)

    with _isolate_memories_dir([{"title": "rule", "content": "x"}]):
        assert fake_memories.exists()
        assert len(list(fake_memories.glob("*.md"))) == 1

    # After exit: empty directory exists, no fixture leftovers
    assert fake_memories.exists()
    assert not list(fake_memories.glob("*.md"))


# ---------------------------------------------------------------------------
# Layer 2: Fixture YAML lint
# ---------------------------------------------------------------------------


def _all_fixtures() -> list[Path]:
    scenarios_dir = REPO / "evaluation" / "scenarios"
    return sorted(scenarios_dir.rglob("*.yaml"))


@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.stem)
def test_fixture_lint(fixture_path):
    """Each YAML fixture must parse and have the required schema fields.

    Two fixture shapes are accepted:
      1. Single-run: `expected:` at top level.
      2. Variants: `variants:` list, each entry with its own `expected:`.
    """
    import yaml
    data = yaml.safe_load(fixture_path.read_text())
    assert isinstance(data, dict), f"{fixture_path.name}: top-level must be a dict"
    assert "name" in data, f"{fixture_path.name}: missing 'name'"
    assert "inputs" in data, f"{fixture_path.name}: missing 'inputs'"

    has_expected = "expected" in data
    has_variants = "variants" in data
    assert has_expected or has_variants, (
        f"{fixture_path.name}: needs either top-level 'expected' or 'variants' block"
    )

    inputs = data["inputs"]
    for required in ("age", "resources", "detected_entities"):
        assert required in inputs, f"{fixture_path.name}: inputs missing {required!r}"

    for e in inputs["detected_entities"]:
        assert "class" in e and "x" in e and "y" in e, (
            f"{fixture_path.name}: entity missing required fields: {e}"
        )

    if has_variants:
        variants = data["variants"]
        assert isinstance(variants, list) and variants, (
            f"{fixture_path.name}: 'variants' must be a non-empty list"
        )
        for variant in variants:
            assert "name" in variant, f"{fixture_path.name}: variant missing 'name'"
            assert "expected" in variant, (
                f"{fixture_path.name} [{variant.get('name', '?')}]: variant missing 'expected'"
            )


# ---------------------------------------------------------------------------
# Layer 3: Live scenario runs (opt-in via --runlive)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.stem)
def test_scenario_runs(fixture_path):
    """Run a real scenario through ClaudeProvider. Requires ANTHROPIC_API_KEY.

    For variant fixtures, fails if ANY variant fails — the failure message
    aggregates per-variant details so all problems surface at once.
    """
    from evaluation.runner import _load_dotenv, run_scenario
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    results = run_scenario(fixture_path)

    if results and all(r.skipped for r in results):
        pytest.skip(results[0].skip_reason)

    failed = [r for r in results if not r.passed and not r.skipped]
    if failed:
        sections = []
        for result in failed:
            sections.append(
                f"\n{result.name} failed:\n  "
                + "\n  ".join(result.failures)
                + f"\n  Reasoning: {result.reasoning[:300]}"
            )
        pytest.fail("\n".join(sections))
