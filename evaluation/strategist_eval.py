"""Vision regression harness for the strategist's screenshot OCR.

Compares the strategist's resource readings (food/wood/gold/stone/population/age)
against hand-labeled ground truth on captured screenshots. Catches:
  - Model regressions: e.g. Sonnet 4.7 misreading the population indicator
  - Prompt regressions: e.g. a strategist.md edit losing OCR calibration
  - Vision-stack drift: changes elsewhere that disturb the image pipeline

Vision fixtures live in `evaluation/vision_fixtures/*.yaml` and pair a
screenshot path with a tolerance-based assertion block. Tolerance ranges
matter because the strategist's readings are LLM-OCR (noisy by nature):
food=200 ± 20 is meaningful; food == 200 exactly is brittle.

Fixture schema:
    name: dark_age_starting_state
    screenshot: dark_age_200food.png      # relative to fixture file
    expected:
      food: {min: 180, max: 220}           # range = tolerance check
      wood: 150                            # bare int = exact match
      population: "8/15"                   # string = exact match
      age: "Dark Age"

Usage (live):
    pytest tests/test_strategist_vision.py --runlive

Usage (offline DSL tests, no API):
    pytest tests/test_strategist_vision.py
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VISION_FIXTURES_DIR = Path(__file__).resolve().parent / "vision_fixtures"


@dataclass
class VisionResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    actual: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Assertion DSL — independent of LLM, fully testable offline
# ---------------------------------------------------------------------------

def evaluate_resource_readings(expected: dict, actual: dict) -> list[str]:
    """Compare strategist readings against expected, returning failure messages.

    Each expected[field] is one of:
      - {"min": N, "max": M}  → range check (inclusive)
      - {"min": N}            → lower-bound only
      - {"max": M}            → upper-bound only
      - bare value            → exact-match check

    String fields (age, population) only support exact match — ranges don't
    apply. Any field not in `expected` is skipped (not a failure).
    """
    failures: list[str] = []
    for field_name, spec in expected.items():
        if field_name not in actual:
            failures.append(f"missing field {field_name!r} in strategist output")
            continue
        actual_value = actual[field_name]

        if isinstance(spec, dict):
            if not isinstance(actual_value, (int, float)) or isinstance(actual_value, bool):
                # Range specs only make sense for numeric readings. Catch type
                # drift (e.g. strategist returns "200" instead of 200) with a
                # clear message rather than a TypeError from the comparison.
                failures.append(
                    f"{field_name}={actual_value!r} is non-numeric; expected numeric "
                    f"value for range spec {spec!r}"
                )
                continue
            if "min" in spec and actual_value < spec["min"]:
                failures.append(
                    f"{field_name}={actual_value} below min={spec['min']}"
                )
            if "max" in spec and actual_value > spec["max"]:
                failures.append(
                    f"{field_name}={actual_value} above max={spec['max']}"
                )
        else:
            if actual_value != spec:
                failures.append(
                    f"{field_name}={actual_value!r} != expected {spec!r}"
                )
    return failures


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_vision_fixture(fixture_path: Path) -> dict:
    """Load and validate a vision fixture YAML."""
    data = yaml.safe_load(fixture_path.read_text()) or {}
    if "name" not in data:
        raise ValueError(f"{fixture_path}: missing 'name'")
    if "screenshot" not in data:
        raise ValueError(f"{fixture_path}: missing 'screenshot'")
    if "expected" not in data:
        raise ValueError(f"{fixture_path}: missing 'expected'")
    return data


def resolve_screenshot_path(fixture_path: Path, screenshot_field: str) -> Path:
    """Screenshots are relative to the fixture file by default."""
    candidate = Path(screenshot_field)
    if candidate.is_absolute():
        return candidate
    return fixture_path.parent / candidate


def all_vision_fixtures() -> list[Path]:
    if not VISION_FIXTURES_DIR.exists():
        return []
    return sorted(VISION_FIXTURES_DIR.rglob("*.yaml"))


# ---------------------------------------------------------------------------
# Live strategist invocation
# ---------------------------------------------------------------------------

async def _invoke_strategist_async(screenshot_bytes: bytes,
                                   model: str | None = None) -> dict:
    """Call StrategistProvider against a screenshot, return its resource_readings dict.

    A minimal `GameState` is supplied — only `current_age` matters for the
    prompt's text portion; the strategist's job is to RE-READ those numbers
    from the image, so the seed values shouldn't bias the OCR.
    """
    from gameplay_agent.memory import GameState
    from gameplay_agent.providers.strategist import StrategistProvider

    provider = StrategistProvider(model=model)
    game_state = GameState()  # all defaults; strategist re-reads from screenshot

    try:
        _, readings = await provider.generate_goals(
            game_state,
            current_goals_summary="",
            detected_entities_summary="",
            turn=1,
            screenshot_bytes=screenshot_bytes,
        )
        return readings or {}
    finally:
        with contextlib.suppress(Exception):
            await provider.client.close()


def run_vision_check(fixture_path: Path, *, model: str | None = None) -> VisionResult:
    """Run a single vision fixture: load image → call strategist → compare."""
    fixture = load_vision_fixture(fixture_path)
    name = fixture["name"]
    screenshot_path = resolve_screenshot_path(fixture_path, fixture["screenshot"])

    if not screenshot_path.exists():
        return VisionResult(
            name=name, passed=False,
            failures=[f"screenshot not found: {screenshot_path}"],
        )

    screenshot_bytes = screenshot_path.read_bytes()

    try:
        readings = asyncio.run(_invoke_strategist_async(screenshot_bytes, model=model))
    except Exception as exc:
        return VisionResult(
            name=name, passed=False,
            failures=[f"strategist invocation failed: {type(exc).__name__}: {exc}"],
        )

    failures = evaluate_resource_readings(fixture["expected"], readings)
    return VisionResult(
        name=name,
        passed=not failures,
        failures=failures,
        actual=readings,
    )
