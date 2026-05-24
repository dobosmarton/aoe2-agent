"""Scenario runner: load YAML fixture → run executor → evaluate assertions.

Reuses the production `ClaudeProvider.get_actions()` so the test path matches
real gameplay exactly. The only thing mocked is `execute_action` (so the
agentic tool loop runs without pyautogui side effects).

CLI:
    python -m gameplay_agent.scenario_runner gameplay_agent/scenarios/age_up_gate_fires.yaml
    python -m gameplay_agent.scenario_runner gameplay_agent/scenarios/*.yaml
    python -m gameplay_agent.scenario_runner --all
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.world_sim import (
    WorldState,
    apply_actions,
    evaluate_end_state,
    init_from_fixture,
    state_to_fixture_inputs,
    tick,
)
from gameplay_agent.assertions import evaluate, matches
from gameplay_agent.context_builder import _build_context
from gameplay_agent.test_isolation import (
    _isolate_memories_dir,
    _mock_executor,
    _seed_detected_entities,
)

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants — named so a future reader doesn't have to guess
# ---------------------------------------------------------------------------

DEFAULT_GAME_WIDTH = 1920
DEFAULT_GAME_HEIGHT = 1080
RECENT_TURNS_CONTEXT_WINDOW = 3
COST_DECIMAL_PLACES = 4
SUMMARY_SEPARATOR_WIDTH = 60

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Tiny .env loader (no python-dotenv dep)
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    actions: list[dict] = field(default_factory=list)
    reasoning: str = ""
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def _load_fixture(fixture_path: Path) -> dict:
    import yaml

    return yaml.safe_load(fixture_path.read_text()) or {}


def _is_real_screenshot_scenario(fixture: dict) -> bool:
    return bool(fixture.get("screenshot"))


async def _invoke_executor(fixture: dict, model: str | None) -> tuple[list[dict], str, float]:
    """Run the production executor against the fixture context.

    Returns (executed_actions, reasoning, cost_usd). The memory + executor
    mocks are managed by the caller's `with` statements.
    """
    from gameplay_agent.providers.claude import ClaudeProvider

    provider = ClaudeProvider(model=model)
    _seed_detected_entities(fixture.get("inputs", {}).get("detected_entities", []))
    context = _build_context(fixture)

    try:
        response = await provider.get_actions(
            context,
            width=DEFAULT_GAME_WIDTH,
            height=DEFAULT_GAME_HEIGHT,
        )
        return (
            response.get("actions", []),
            response.get("reasoning", ""),
            round(provider._cumulative_cost_usd(), COST_DECIMAL_PLACES),
        )
    finally:
        # AsyncAnthropic owns an httpx pool; close it so connections don't
        # leak across scenarios in the shared event loop.
        with contextlib.suppress(Exception):
            await provider.client.close()


_VARIANT_OVERRIDABLE_INPUTS = ("memories", "strategist_overrides")


def _expand_variants(fixture: dict) -> list[dict]:
    """Return one fixture-per-variant. No `variants:` key = single anonymous run.

    Each returned fixture has `_variant_name` set (None for non-variant fixtures).
    Variants can override these `inputs:` keys: `memories`, `strategist_overrides`.
    Each variant's `expected:` block is its own; falls back to the top-level one
    if the variant doesn't supply its own.
    """
    if "variants" not in fixture:
        return [{**fixture, "_variant_name": None}]

    base_inputs = fixture.get("inputs", {})
    expanded: list[dict] = []
    for index, variant in enumerate(fixture["variants"]):
        variant_inputs = {**base_inputs}
        for key in _VARIANT_OVERRIDABLE_INPUTS:
            if key in variant:
                variant_inputs[key] = variant[key]
        expanded.append(
            {
                **fixture,
                "inputs": variant_inputs,
                "expected": variant.get("expected", fixture.get("expected", {})),
                "_variant_name": variant.get("name", f"variant_{index}"),
            }
        )
    return expanded


def _scenario_display_name(fixture_path: Path, variant_name: str | None) -> str:
    base = fixture_path.stem
    return f"{base} [{variant_name}]" if variant_name else base


async def _run_one_variant_async(
    fixture: dict,
    fixture_path: Path,
    *,
    model: str | None = None,
    baseline_actions: list[dict] | None = None,
) -> ScenarioResult:
    """Run a single (possibly variant-overlaid) fixture through the executor.

    `baseline_actions` is the first variant's executed actions, threaded
    through to `evaluate()` so `differs_from_baseline_by` assertions can
    compare against it. None for the baseline variant itself.
    """
    name = _scenario_display_name(fixture_path, fixture.get("_variant_name"))

    if _is_real_screenshot_scenario(fixture):
        return ScenarioResult(
            name=name,
            passed=True,
            skipped=True,
            skip_reason="real-screenshot scenarios not yet supported in v1 runner",
        )

    inputs = fixture.get("inputs", {})
    expected = fixture.get("expected", {})
    fixture_memories = inputs.get("memories", [])
    started = time.monotonic()

    with _isolate_memories_dir(fixture_memories), _mock_executor():
        try:
            actions, reasoning, cost = await _invoke_executor(fixture, model)
        except Exception as exc:
            return ScenarioResult(
                name=name,
                passed=False,
                failures=[f"runner exception: {type(exc).__name__}: {exc}"],
                duration_s=time.monotonic() - started,
            )

    failures = (
        evaluate(expected, actions=actions, reasoning=reasoning, baseline_actions=baseline_actions)
        if expected
        else []
    )
    return ScenarioResult(
        name=name,
        passed=(not failures),
        failures=failures,
        cost_usd=cost,
        duration_s=time.monotonic() - started,
        actions=actions,
        reasoning=reasoning,
    )


@dataclass
class _MultiTurnConfig:
    """Parsed `multi_turn:` section of a scenario fixture."""

    max_turns: int
    per_turn_expected: dict
    end_state_spec: dict
    eventually_pattern: dict | None

    @classmethod
    def from_fixture(cls, fixture: dict) -> _MultiTurnConfig:
        cfg = fixture["multi_turn"]
        return cls(
            max_turns=int(cfg.get("max_turns", 10)),
            per_turn_expected=cfg.get("expected", {}),
            end_state_spec=cfg.get("end_state", {}),
            eventually_pattern=cfg.get("eventually_includes"),
        )


async def _run_multi_turn_step(
    fixture: dict,
    base_inputs: dict,
    world_state: WorldState,
    recent_turns: list[dict],
    turn_num: int,
    per_turn_expected: dict,
    model: str | None,
) -> tuple[WorldState, list[dict], str, float, list[str]]:
    """Build context → invoke executor → assert → apply → tick. Returns (new_state, actions, reasoning, cost, per_turn_failures)."""
    current_inputs = state_to_fixture_inputs(world_state, base_inputs)
    current_inputs = {
        **current_inputs,
        "recent_turns": recent_turns[-RECENT_TURNS_CONTEXT_WINDOW:],
    }
    current_fixture = {**fixture, "inputs": current_inputs}

    actions, reasoning, cost = await _invoke_executor(current_fixture, model)

    failures: list[str] = []
    if per_turn_expected:
        failures.extend(
            f"turn {turn_num}: {f}"
            for f in evaluate(per_turn_expected, actions=actions, reasoning=reasoning)
        )

    new_state = tick(apply_actions(world_state, actions))
    return new_state, actions, reasoning, cost, failures


def _evaluate_multi_turn_end(
    cfg: _MultiTurnConfig,
    world_state: WorldState,
    all_actions: list[dict],
) -> list[str]:
    """Aggregate end-of-run assertions: end_state and eventually_includes."""
    failures: list[str] = []
    if cfg.end_state_spec:
        failures.extend(evaluate_end_state(cfg.end_state_spec, world_state))
    if cfg.eventually_pattern is not None and not any(
        matches(a, cfg.eventually_pattern) for a in all_actions
    ):
        failures.append(
            f"eventually_includes FAILED — no turn produced an action matching "
            f"{cfg.eventually_pattern!r} across {cfg.max_turns} turns"
        )
    return failures


async def _run_multi_turn_scenario_async(
    fixture: dict,
    fixture_path: Path,
    *,
    model: str | None = None,
) -> list[ScenarioResult]:
    """Run a multi-turn scenario through the world simulator.

    Each turn: tick world state → build context → invoke executor → apply actions.
    The world simulator evolves resources, population, and age across N turns
    without booting the real game. Only `execute_action` is mocked.

    Fixture schema (under `multi_turn:`):
      max_turns: int          — how many turns to run (default 10)
      end_state:              — WorldState fields to assert after the final turn
        age: "Feudal Age"       (string → exact equality)
        population: 15          (int/float → ≥ semantics)
      eventually_includes:    — action pattern that must appear in ANY turn
        type: press
        key: z
      expected:               — assertion block applied to EACH turn's actions
        must_not_include: {type: press, key: b}
    """
    name = fixture_path.stem
    cfg = _MultiTurnConfig.from_fixture(fixture)
    base_inputs = fixture.get("inputs", {})
    world_state = init_from_fixture(base_inputs)
    fixture_memories = base_inputs.get("memories", [])

    all_actions: list[dict] = []
    all_failures: list[str] = []
    recent_turns: list[dict] = []
    total_cost = 0.0
    started = time.monotonic()

    with _isolate_memories_dir(fixture_memories), _mock_executor():
        for turn_num in range(1, cfg.max_turns + 1):
            try:
                world_state, actions, reasoning, cost, step_failures = await _run_multi_turn_step(
                    fixture,
                    base_inputs,
                    world_state,
                    recent_turns,
                    turn_num,
                    cfg.per_turn_expected,
                    model,
                )
            except Exception as exc:
                # Executor crashed mid-run: stop the loop but still evaluate the
                # end-state spec against the partial world below — the assertion
                # is "what did the agent build before it crashed", not just
                # "did the agent crash".
                all_failures.append(
                    f"turn {turn_num}: runner exception: {type(exc).__name__}: {exc}"
                )
                break

            total_cost += cost
            all_actions.extend(actions)
            all_failures.extend(step_failures)
            recent_turns.append({"iteration": turn_num, "reasoning": reasoning})

    all_failures.extend(_evaluate_multi_turn_end(cfg, world_state, all_actions))

    return [
        ScenarioResult(
            name=name,
            passed=not all_failures,
            failures=all_failures,
            cost_usd=round(total_cost, COST_DECIMAL_PLACES),
            duration_s=time.monotonic() - started,
            actions=all_actions,
            reasoning=f"(multi-turn: {world_state.turn} turns run)",
        )
    ]


async def _run_scenario_async(
    fixture_path: Path,
    *,
    model: str | None = None,
) -> list[ScenarioResult]:
    """Run all variants of a scenario. Returns one ScenarioResult per variant.

    Multi-turn fixtures (containing `multi_turn:` key) are handled by
    `_run_multi_turn_scenario_async`; everything else goes through the
    single-turn variant path.

    The first variant's executed actions become the baseline for any
    subsequent variants that use `differs_from_baseline_by`.
    Non-variant fixtures produce a single-element list with no baseline.
    """
    fixture = _load_fixture(fixture_path)

    if "multi_turn" in fixture:
        return await _run_multi_turn_scenario_async(fixture, fixture_path, model=model)

    variants = _expand_variants(fixture)
    results: list[ScenarioResult] = []
    baseline_actions: list[dict] | None = None
    for index, variant_fixture in enumerate(variants):
        result = await _run_one_variant_async(
            variant_fixture,
            fixture_path,
            model=model,
            baseline_actions=baseline_actions,
        )
        results.append(result)
        if index == 0 and not result.skipped:
            baseline_actions = result.actions
    return results


async def _run_all_async(
    fixtures: list[Path],
    *,
    model: str | None,
    on_each: Callable[[ScenarioResult], None] = lambda result: None,
) -> list[ScenarioResult]:
    """Run every scenario (possibly multi-variant) in a SINGLE shared event loop.

    Each variant gets its own ClaudeProvider (so memories are correctly
    isolated) but they share the asyncio loop, which prevents httpx
    transport cleanup from racing against a closed loop.
    """
    results: list[ScenarioResult] = []
    for path in fixtures:
        for result in await _run_scenario_async(path, model=model):
            results.append(result)
            on_each(result)
    return results


def run_scenario(fixture_path: Path, *, model: str | None = None) -> list[ScenarioResult]:
    """Synchronous entry point for one-off use (e.g. pytest).

    Returns a list — one ScenarioResult per variant, or a single-element
    list for non-variant fixtures.
    """
    return asyncio.run(_run_scenario_async(fixture_path, model=model))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_result(result: ScenarioResult) -> str:
    if result.skipped:
        return f"  ⚪ {result.name}  SKIPPED — {result.skip_reason}"

    color = ANSI_GREEN if result.passed else ANSI_RED
    status = "✓ PASS" if result.passed else "✗ FAIL"
    header = (
        f"  {color}{status}{ANSI_RESET}  {result.name}  "
        f"({result.duration_s:.1f}s, ${result.cost_usd:.4f}, {len(result.actions)} actions)"
    )
    if not result.failures:
        return header
    failure_lines = "\n".join(
        "      " + line for failure in result.failures for line in failure.splitlines()
    )
    return f"{header}\n{failure_lines}"


def _print_summary(results: list[ScenarioResult]) -> None:
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.duration_s for r in results)

    separator = "=" * SUMMARY_SEPARATOR_WIDTH
    print()
    print(separator)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"  ${total_cost:.4f} total cost, {total_time:.1f}s wall-clock")
    print(separator)


class _RunnerArgs(argparse.Namespace):
    fixtures: list[str]
    all: bool
    model: str | None


_SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def _resolve_fixtures(args: _RunnerArgs) -> list[Path]:
    if args.all:
        return sorted(_SCENARIOS_DIR.rglob("*.yaml"))
    return [Path(p) for p in args.fixtures]


def _parse_args() -> _RunnerArgs:
    parser = argparse.ArgumentParser(description="Run scenario evaluations against ClaudeProvider")
    parser.add_argument("fixtures", nargs="*", help="YAML fixture paths (or use --all)")
    parser.add_argument(
        "--all", action="store_true", help="Run every fixture in gameplay_agent/scenarios/"
    )
    parser.add_argument("--model", help="Override the model (default: config.model)")
    return parser.parse_args(namespace=_RunnerArgs())


def main() -> int:
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (checked env + .env file).")
        return 1

    args = _parse_args()
    fixtures = _resolve_fixtures(args)
    if not fixtures:
        print("No fixtures specified. Use --all or pass YAML paths.")
        return 1

    valid_fixtures: list[Path] = []
    for fixture_path in fixtures:
        if fixture_path.exists():
            valid_fixtures.append(fixture_path)
        else:
            print(f"  ⚠ {fixture_path}: not found")

    print(f"Running {len(valid_fixtures)} scenario(s)...\n")
    results = asyncio.run(
        _run_all_async(
            valid_fixtures,
            model=args.model,
            on_each=lambda result: print(_format_result(result)),
        )
    )
    _print_summary(results)

    return 0 if all(r.passed or r.skipped for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
