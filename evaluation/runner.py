"""Scenario runner: load YAML fixture → run executor → evaluate assertions.

Reuses the production `ClaudeProvider.get_actions()` so the test path matches
real gameplay exactly. The only thing mocked is `execute_action` (so the
agentic tool loop runs without pyautogui side effects).

CLI:
    python -m evaluation.runner evaluation/scenarios/age_up_gate_fires.yaml
    python -m evaluation.runner evaluation/scenarios/*.yaml
    python -m evaluation.runner --all
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Sibling modules — must be imported AFTER the sys.path mutation above so a
# direct `python evaluation/runner.py` invocation can resolve the package.
from evaluation.assertions import evaluate, matches  # noqa: E402
from evaluation.world_sim import (  # noqa: E402
    WorldState,
    apply_actions,
    evaluate_end_state,
    init_from_fixture,
    state_to_fixture_inputs,
    tick,
)

# ---------------------------------------------------------------------------
# Constants — named so a future reader doesn't have to guess
# ---------------------------------------------------------------------------

DEFAULT_GAME_WIDTH = 1920
DEFAULT_GAME_HEIGHT = 1080
ENTITY_BBOX_HALF_SIZE = 20
DEFAULT_ENTITY_CONFIDENCE = 0.9
PRIORITY_HIGH_THRESHOLD = 8
PRIORITY_MED_THRESHOLD = 5
RECENT_TURN_REASONING_PREVIEW = 100
RECENT_TURNS_CONTEXT_WINDOW = 3
COST_DECIMAL_PLACES = 4
SUMMARY_SEPARATOR_WIDTH = 60

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

DEFAULT_AGE = "Dark Age"


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
# Memory directory backup/restore + fixture planting
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _isolate_memories_dir(fixture_memories: list[dict]):
    """Back up existing memories/, plant fixture memories, restore on exit.

    Refuses to run if an orphan `<memories>_eval_backup` directory exists
    from a crashed prior run — its contents are the user's real memories
    and we cannot tell which of the two dirs is canonical without asking.
    """
    from autoresearch.memory_chain import MEMORIES_DIR

    backup_dir = MEMORIES_DIR.with_name(MEMORIES_DIR.name + "_eval_backup")
    if backup_dir.exists():
        raise RuntimeError(
            f"Found orphan eval backup at {backup_dir}. A prior evaluation "
            f"run crashed before restoring your real memories. Inspect both "
            f"{backup_dir} and {MEMORIES_DIR}, move the canonical contents "
            f"back to {MEMORIES_DIR}, then delete the other. Refusing to "
            f"proceed to avoid silent data loss."
        )

    had_existing = MEMORIES_DIR.exists()
    if had_existing:
        shutil.move(str(MEMORIES_DIR), str(backup_dir))
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

    for index, memory in enumerate(fixture_memories, start=1):
        _write_fixture_memory(MEMORIES_DIR, memory, index)

    try:
        yield
    finally:
        shutil.rmtree(MEMORIES_DIR, ignore_errors=True)
        if had_existing and backup_dir.exists():
            shutil.move(str(backup_dir), str(MEMORIES_DIR))
        else:
            MEMORIES_DIR.mkdir(parents=True, exist_ok=True)


def _write_fixture_memory(memories_dir: Path, memory: dict, index: int) -> None:
    """Write a single fixture memory file with frontmatter."""
    title = memory.get("title", f"fixture_memory_{index}")
    applies_when = memory.get("applies_when", "any")
    score_impact = memory.get("score_impact", "negative")
    mem_type = memory.get("type", "economy")
    content = memory.get("content", "I should follow this rule.")
    path = memories_dir / f"{index:03d}_{title}.md"
    path.write_text(
        f"---\n"
        f"type: {mem_type}\n"
        f"title: {title}\n"
        f"game_id: fixture\n"
        f"applies_when: {applies_when}\n"
        f"score_impact: {score_impact}\n"
        f"created: 2026-04-25T00:00:00+00:00\n"
        f"---\n\n{content}\n"
    )


# ---------------------------------------------------------------------------
# Executor mocking — patch execute_action so the agentic loop doesn't click
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mock_executor():
    """Patch execute_action in both the canonical module and the import in claude.py.

    The executor's tool loop calls execute_action for every action; without
    mocking it would invoke pyautogui (real clicks). We replace it with a
    success-returning no-op so the LLM's behavior loop is preserved.
    """
    import gameplay_agent.executor as ex
    import gameplay_agent.providers.claude as claude_mod

    real_canonical = ex.execute_action
    real_in_claude = claude_mod.execute_action

    async def fake_execute_action(action_dict):
        return ex.ActionResult(success=True, detail="ok (eval)")

    ex.execute_action = fake_execute_action
    claude_mod.execute_action = fake_execute_action

    try:
        yield
    finally:
        ex.execute_action = real_canonical
        claude_mod.execute_action = real_in_claude


# ---------------------------------------------------------------------------
# Build inputs from a fixture (synthetic detection list + context string)
# ---------------------------------------------------------------------------


def _seed_detected_entities(entities: list[dict]) -> None:
    """Push fixture entities into executor module state so target_class resolution works."""
    import gameplay_agent.executor as ex

    ex._detected_entities = [_entity_dict(entity, index) for index, entity in enumerate(entities)]


def _entity_dict(entity: dict, index: int) -> dict:
    """Convert a fixture entity into the executor's internal dict shape."""
    class_name = entity.get("class", "unknown")
    x = entity.get("x", 0)
    y = entity.get("y", 0)
    return {
        "id": entity.get("id", f"{class_name}_{index}"),
        "class": class_name,
        "center": (x, y),
        "bbox": (
            x - ENTITY_BBOX_HALF_SIZE,
            y - ENTITY_BBOX_HALF_SIZE,
            x + ENTITY_BBOX_HALF_SIZE,
            y + ENTITY_BBOX_HALF_SIZE,
        ),
        "confidence": entity.get("confidence", DEFAULT_ENTITY_CONFIDENCE),
    }


def _format_entity_line(entity: dict, index: int) -> str:
    class_name = entity.get("class", "unknown")
    x = int(entity.get("x", 0))
    y = int(entity.get("y", 0))
    confidence = float(entity.get("confidence", DEFAULT_ENTITY_CONFIDENCE))
    entity_id = entity.get("id", f"{class_name}_{index}")
    return f"  {entity_id}: {class_name} at ({x},{y}) [{confidence:.0%}]"


def _build_entity_summary(entities: list[dict]) -> str:
    """Mirror gameplay_agent.entity_utils.build_entity_summary's output shape."""
    if not entities:
        return ""
    return "\n".join(_format_entity_line(entity, index) for index, entity in enumerate(entities))


def _priority_tier(priority: int) -> str:
    if priority >= PRIORITY_HIGH_THRESHOLD:
        return "HIGH"
    if priority >= PRIORITY_MED_THRESHOLD:
        return "MED"
    return "LOW"


def _build_resource_block(resources: dict, age: str) -> str:
    return "\n".join(
        [
            "## Resource Status (from strategist)",
            f"- Food: {resources.get('food', '?')}",
            f"- Wood: {resources.get('wood', '?')}",
            f"- Gold: {resources.get('gold', '?')}",
            f"- Stone: {resources.get('stone', '?')}",
            f"- Population: {resources.get('population', '0/0')}",
            f"- Age: {age}",
        ]
    )


def _build_goal_block(goals: list[dict]) -> str:
    if not goals:
        return ""
    lines = ["## Active Goals"]
    for goal in goals:
        priority = goal.get("priority", PRIORITY_MED_THRESHOLD)
        tier = _priority_tier(priority)
        lines.append(
            f"  {tier} (P{priority}): {goal.get('name', '?')} → "
            f"{goal.get('metric')} target {goal.get('target')}"
        )
    return "\n".join(lines)


def _build_state_block(resources: dict, age: str, under_attack: bool) -> str:
    population = resources.get("population", "0/0")
    pop_now, _, pop_cap = population.partition("/")
    non_pop_resources = " ".join(
        f"{key}={value}" for key, value in resources.items() if key != "population"
    )
    lines = [
        "## Current Game State",
        f"- Resources: {non_pop_resources}",
        f"- Population: {pop_now or 0}/{pop_cap or 0}",
        f"- Age: {age}",
    ]
    if under_attack:
        lines.append("- under_attack: true")
    return "\n".join(lines)


def _build_entity_block(entities: list[dict]) -> str:
    summary = _build_entity_summary(entities)
    if not summary:
        return ""
    return (
        "\n## Detected Entities (from YOLO)\n"
        "Use target_class or target_id to interact with these:\n"
        f"{summary}\n"
    )


def _build_recent_turns_block(recent_turns: list[dict]) -> str:
    if not recent_turns:
        return ""
    lines = ["## Recent Turns (last 3)"]
    for turn in recent_turns:
        iteration = turn.get("iteration", "?")
        reasoning_preview = turn.get("reasoning", "")[:RECENT_TURN_REASONING_PREVIEW]
        lines.append(f"Turn {iteration}: {reasoning_preview}")
    return "\n".join(lines)


def _apply_strategist_overrides(inputs: dict) -> dict:
    """Merge `strategist_overrides:` on top of base inputs.

    The strategist normally provides resource readings and goals to the
    executor. This helper lets a fixture express "what if the strategist
    output something different?" without running the real strategist.

    Merge semantics:
      resources  — shallow merge (override individual fields, preserve others)
      goals      — replace entirely (lists have no canonical partial-merge)

    Returns a new dict; the original is never mutated.
    """
    overrides = inputs.get("strategist_overrides") or {}
    if not overrides:
        return inputs

    merged = {**inputs}
    if "resources" in overrides:
        merged["resources"] = {
            **inputs.get("resources", {}),
            **overrides["resources"],
        }
    if "goals" in overrides:
        merged["goals"] = overrides["goals"]
    return merged


def _build_context(fixture: dict) -> str:
    """Assemble the context string the same way game_loop._build_llm_context does.

    Order matches the production assembly: entities → goals → resources → state → recent.
    """
    inputs = fixture.get("inputs", {})
    inputs = _apply_strategist_overrides(inputs)
    resources = inputs.get("resources", {})
    age = inputs.get("age", DEFAULT_AGE)

    blocks = [
        _build_entity_block(inputs.get("detected_entities", [])),
        _build_goal_block(inputs.get("goals", [])),
        _build_resource_block(resources, age),
        _build_state_block(resources, age, bool(inputs.get("under_attack"))),
        _build_recent_turns_block(inputs.get("recent_turns", [])),
    ]
    return "\n\n".join(block for block in blocks if block)


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


def _resolve_fixtures(args: argparse.Namespace) -> list[Path]:
    if args.all:
        scenarios_dir = REPO / "evaluation" / "scenarios"
        return sorted(scenarios_dir.rglob("*.yaml"))
    return [Path(p) for p in args.fixtures]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario evaluations against ClaudeProvider")
    parser.add_argument("fixtures", nargs="*", help="YAML fixture paths (or use --all)")
    parser.add_argument(
        "--all", action="store_true", help="Run every fixture in evaluation/scenarios/"
    )
    parser.add_argument("--model", help="Override the model (default: config.model)")
    return parser.parse_args()


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
