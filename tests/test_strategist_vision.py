"""Vision regression tests for the strategist's screenshot OCR.

Three layers (mirrors test_evaluation.py):
  1. Assertion DSL unit tests (no LLM, no screenshots, fast)
  2. Vision fixture YAML lint (no LLM, fast)
  3. Live strategist runs against labeled screenshots (requires
     ANTHROPIC_API_KEY and labeled fixtures; gated by --runlive)

Add new vision fixtures by:
  1. Drop a screenshot into evaluation/vision_fixtures/screenshots/
  2. Create evaluation/vision_fixtures/<name>.yaml referencing it
  3. Run: pytest tests/test_strategist_vision.py --runlive -k <name>
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

def test_evaluate_range_match_passes():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": {"min": 180, "max": 220}}
    actual = {"food": 200, "wood": 0}
    assert evaluate_resource_readings(expected, actual) == []


def test_evaluate_range_below_min_fails():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": {"min": 180, "max": 220}}
    actual = {"food": 150}
    failures = evaluate_resource_readings(expected, actual)
    assert any("below min=180" in f for f in failures)


def test_evaluate_range_above_max_fails():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": {"min": 180, "max": 220}}
    actual = {"food": 250}
    failures = evaluate_resource_readings(expected, actual)
    assert any("above max=220" in f for f in failures)


def test_evaluate_lower_bound_only():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": {"min": 100}}
    assert evaluate_resource_readings(expected, {"food": 100}) == []
    assert evaluate_resource_readings(expected, {"food": 99}) != []
    assert evaluate_resource_readings(expected, {"food": 10000}) == []


def test_evaluate_upper_bound_only():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": {"max": 500}}
    assert evaluate_resource_readings(expected, {"food": 500}) == []
    assert evaluate_resource_readings(expected, {"food": 501}) != []
    assert evaluate_resource_readings(expected, {"food": 0}) == []


def test_evaluate_exact_match_pass():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"age": "Dark Age", "population": "8/15"}
    actual = {"age": "Dark Age", "population": "8/15"}
    assert evaluate_resource_readings(expected, actual) == []


def test_evaluate_exact_match_fail():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"age": "Dark Age"}
    actual = {"age": "Feudal Age"}
    failures = evaluate_resource_readings(expected, actual)
    assert any("'Feudal Age' != expected 'Dark Age'" in f for f in failures)


def test_evaluate_missing_field_fails():
    """Strategist output should always include every standard field; missing = bug."""
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": 200}
    actual = {"wood": 100}  # food missing entirely
    failures = evaluate_resource_readings(expected, actual)
    assert any("missing field 'food'" in f for f in failures)


def test_evaluate_extra_actual_fields_ok():
    """Actual readings can have more fields than expected — only listed ones matter."""
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {"food": 200}
    actual = {"food": 200, "wood": 100, "gold": 50, "stone": 25}
    assert evaluate_resource_readings(expected, actual) == []


def test_evaluate_combined_range_and_exact():
    from evaluation.strategist_eval import evaluate_resource_readings
    expected = {
        "food": {"min": 180, "max": 220},
        "age": "Dark Age",
        "population": "8/15",
    }
    actual = {"food": 195, "age": "Dark Age", "population": "8/15"}
    assert evaluate_resource_readings(expected, actual) == []


# ---------------------------------------------------------------------------
# Layer 2: Vision fixture path resolution + YAML lint
# ---------------------------------------------------------------------------

def test_resolve_screenshot_path_relative():
    from evaluation.strategist_eval import resolve_screenshot_path
    fixture_path = Path("/tmp/fixtures/dark_age.yaml")
    resolved = resolve_screenshot_path(fixture_path, "screenshots/img.png")
    assert resolved == Path("/tmp/fixtures/screenshots/img.png")


def test_resolve_screenshot_path_absolute():
    from evaluation.strategist_eval import resolve_screenshot_path
    fixture_path = Path("/tmp/fixtures/dark_age.yaml")
    resolved = resolve_screenshot_path(fixture_path, "/abs/path/img.png")
    assert resolved == Path("/abs/path/img.png")


def test_load_vision_fixture_validates_required_fields(tmp_path):
    from evaluation.strategist_eval import load_vision_fixture
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nscreenshot: foo.png\n")  # missing `expected`
    with pytest.raises(ValueError, match="missing 'expected'"):
        load_vision_fixture(bad)


def test_load_vision_fixture_round_trip(tmp_path):
    """A well-formed fixture loads cleanly with all required keys."""
    from evaluation.strategist_eval import load_vision_fixture
    good = tmp_path / "good.yaml"
    good.write_text(
        "name: test_fixture\n"
        "screenshot: img.png\n"
        "expected:\n"
        "  food: 200\n"
        "  age: \"Dark Age\"\n"
    )
    data = load_vision_fixture(good)
    assert data["name"] == "test_fixture"
    assert data["screenshot"] == "img.png"
    assert data["expected"]["food"] == 200


def _all_vision_fixtures() -> list[Path]:
    from evaluation.strategist_eval import all_vision_fixtures
    return all_vision_fixtures()


@pytest.mark.parametrize("fixture_path", _all_vision_fixtures(),
                         ids=lambda p: p.stem)
def test_vision_fixture_lint(fixture_path):
    """Every vision fixture must have name, screenshot, expected — and the
    referenced screenshot file must exist."""
    from evaluation.strategist_eval import load_vision_fixture, resolve_screenshot_path
    fixture = load_vision_fixture(fixture_path)
    screenshot_path = resolve_screenshot_path(fixture_path, fixture["screenshot"])
    assert screenshot_path.exists(), (
        f"{fixture_path.name} references missing screenshot {screenshot_path}"
    )


# ---------------------------------------------------------------------------
# Layer 3: Live strategist runs (opt-in via --runlive)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.parametrize("fixture_path", _all_vision_fixtures(),
                         ids=lambda p: p.stem)
def test_strategist_vision_reading(fixture_path):
    """Run the real strategist against a labeled screenshot."""
    from evaluation.runner import _load_dotenv
    from evaluation.strategist_eval import run_vision_check

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = run_vision_check(fixture_path)
    if not result.passed:
        details = "\n  ".join(result.failures)
        pytest.fail(
            f"\n{result.name} vision check failed:\n  {details}\n"
            f"  Actual readings: {result.actual}"
        )
