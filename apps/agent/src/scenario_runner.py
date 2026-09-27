"""Input-driven replay of the production perception → actor → executor path.

Fixtures in ``scenarios/production`` declare an initial observed state and
named action steps. A step's ``after`` observation is applied only when the
expected spending input actually fires. Older prompt-only fixtures belong to
``provider_scenario_runner`` and do not measure gameplay feasibility.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from . import executor
from .entity_snapshot import snapshot_entities
from .goal_logger import GoalLogger
from .goals import GoalManager
from .loops.act import act_once
from .loops.context import LoopContext
from .loops.perceive import perceive_once
from .loops.snapshot import Perception, SpatialRefresh
from .loops.source import GameActuator, Sighting
from .memory import AgentMemory
from .policy.advice import PolicyAdvice, readonly_probabilities

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .policy.advice import PolicyRequest
    from .resource_ocr import ResourceReadings
    from .turn_timing import TickTimings


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    action: str
    after: Mapping[str, object]


@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    observed_age: str = ""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return cast("Mapping[str, object]", value)


def _steps(value: object) -> tuple[ScenarioStep, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("steps must be a non-empty list")
    steps: list[ScenarioStep] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"steps[{index}]")
        action = item.get("action")
        if not isinstance(action, str):
            raise ValueError(f"steps[{index}].action must be a name")
        steps.append(ScenarioStep(action, _mapping(item.get("after", {}), "after")))
    return tuple(steps)


class _ScriptedWorld:
    def __init__(self, initial: Mapping[str, object]) -> None:
        self.resources = {
            name: _integer(_mapping(initial.get("resources", {}), "resources").get(name), name)
            for name in ("food", "wood", "gold", "stone")
        }
        self.population = _integer(initial.get("population"), "population")
        self.population_cap = _integer(initial.get("population_cap"), "population_cap")
        self.villagers = _integer(initial.get("villagers", self.population), "villagers")
        worker_counts = _mapping(initial.get("worker_counts", {}), "worker_counts")
        self.worker_counts = {
            kind: _integer(worker_counts.get(kind, 0), f"{kind}_workers")
            for kind in ("food", "wood", "gold", "stone")
        }
        age = initial.get("age", "Dark Age")
        if not isinstance(age, str):
            raise ValueError("age must be text")
        self.age = age
        self.entities = _entities(initial.get("entities", []))
        self.idle_present = bool(initial.get("idle_present", False))
        self.idle_count = _integer(initial.get("idle_count", 0), "idle_count")

    def apply(self, after: Mapping[str, object], point: tuple[float, float] | None) -> None:
        if "resources" in after:
            for name, value in _mapping(after["resources"], "after.resources").items():
                if name not in self.resources:
                    raise ValueError(f"unknown resource {name}")
                self.resources[name] = _integer(value, name)
        for name in ("population", "population_cap", "idle_count"):
            if name in after:
                setattr(self, name, _integer(after[name], name))
        if "villagers" in after:
            self.villagers = _integer(after["villagers"], "villagers")
        if "worker_counts" in after:
            for kind, value in _mapping(after["worker_counts"], "after.worker_counts").items():
                if kind not in self.worker_counts:
                    raise ValueError(f"unknown worker kind {kind}")
                self.worker_counts[kind] = _integer(value, f"{kind}_workers")
        if "idle_present" in after:
            self.idle_present = bool(after["idle_present"])
        if "age" in after:
            age = after["age"]
            if not isinstance(age, str):
                raise ValueError("after.age must be text")
            self.age = age
        if "building" in after:
            building = after["building"]
            if not isinstance(building, str):
                raise ValueError("after.building must be text")
            center = point or (700.0, 500.0)
            self.entities.append(
                {"id": f"{building}_{len(self.entities)}", "class": building, "center": center}
            )

    def readings(self) -> dict[str, object]:
        return {
            **self.resources,
            "population": f"{self.population}/{self.population_cap}",
            "age": self.age,
            "idle_present": self.idle_present,
            "idle_count": self.idle_count,
            "villagers": self.villagers,
            **{f"{kind}_workers": count for kind, count in self.worker_counts.items()},
        }


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _entities(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("entities must be a list")
    return [dict(_mapping(entity, "entity")) for entity in value]


class _ScriptedSource:
    def __init__(self, world: _ScriptedWorld, ledger: executor.ActionLedger) -> None:
        self.world = world
        self.ledger = ledger

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        with timings.phase("capture"):
            entities = snapshot_entities(self.world.entities)
            executor.set_detected_entities(entities)
            frame = Perception(
                width=1920,
                height=1080,
                entities=entities,
                hud_readings=cast("ResourceReadings", self.world.readings()),
                input_revision=self.ledger.input_revision,
                spatial_valid=True,
                tick=tick,
                captured_at=time.monotonic(),
            )
        return Sighting(frame)

    async def capture_spatial(self, timings: TickTimings) -> SpatialRefresh:
        with timings.phase("capture"):
            executor.set_detected_entities(snapshot_entities(self.world.entities))
            return SpatialRefresh(time.monotonic(), self.ledger.input_revision, True)

    def close(self) -> None:
        pass


class _ScriptedAdvisor:
    def __init__(self, action: str) -> None:
        self.action = action
        self.advertised = False

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.advertised = self.action in {candidate.id for candidate in request.candidates}
        return PolicyAdvice(
            source_tick=request.source_tick,
            source_captured_at=request.source_captured_at,
            model="scripted",
            action_choice=self.action,
            action_confidence=1.0,
            action_probabilities=readonly_probabilities({self.action: 1.0}),
            allocation_focus="balanced",
            allocation_confidence=1.0,
            allocation_probabilities=readonly_probabilities({"balanced": 1.0}),
        )

    async def aclose(self) -> None:
        pass


class _ScriptedInput:
    def __init__(
        self,
        world: _ScriptedWorld,
        ledger: executor.ActionLedger,
        step: ScenarioStep,
    ) -> None:
        self.world = world
        self.ledger = ledger
        self.step = step
        self.calls: list[str] = []
        self.issued = False

    def _record(self, method: str, key: str = "", point: tuple[float, float] | None = None) -> None:
        self.calls.append(f"{method}:{key}" if key else method)
        if method == "press":
            selected: str | None = None
            if key == "h":
                selected = "town_center"
            elif key == ".":
                selected = "villager"
            else:
                spec = executor.BY_ID.get(self.step.action)
                if spec is not None and spec.kind in {"train", "research"} and key == spec.goto_key:
                    selected = next(iter(spec.requires), "town_center")
            if selected is not None:
                self.ledger.selected_unit = selected
                self.ledger.selected_at_revision = self.ledger.input_revision
        elif (
            method == "click"
            and self.step.action.startswith("build_")
            and not self.ledger.pending_placements
        ):
            self.ledger.selected_unit = "villager"
            self.ledger.selected_at_revision = self.ledger.input_revision
        if self.issued:
            return
        action = self.step.action
        spent = (
            (
                method == "click"
                and any(
                    action == f"build_{pending.building_class}"
                    for pending in self.ledger.pending_placements
                )
            )
            or (
                method == "press"
                and any(
                    action
                    in {
                        f"advance_to_{pending.name.removesuffix('_age')}",
                        f"research_{pending.name}",
                    }
                    and key == pending.tech.research_key
                    for pending in self.ledger.pending_research
                )
            )
            or (
                method == "right_click"
                and action.startswith("assign_")
                and self.ledger.pending_assignment is not None
            )
            or (
                method == "press"
                and any(
                    action
                    == ("queue_villager" if pending.unit == "villager" else f"train_{pending.unit}")
                    and key == executor.UNITS[pending.unit].key
                    for pending in self.ledger.pending_training
                )
            )
        )
        if spent:
            self.issued = True
            self.world.apply(self.step.after, point)

    def press(self, key: str) -> None:
        self._record("press", key)

    def hotkey(self, *keys: str) -> None:
        self._record("press", keys[-1])

    def click(self, x: float, y: float) -> None:
        self._record("click", point=(x, y))

    def rightClick(self, x: float, y: float) -> None:
        self._record("right_click", point=(x, y))

    def moveTo(self, *_args: object) -> None:
        pass

    def drag(self, *_args: object, **_kwargs: object) -> None:
        self._record("drag")

    def scroll(self, *_args: object, **_kwargs: object) -> None:
        self._record("scroll")


async def run_scenario_async(path: Path) -> ScenarioResult:
    """Run one scripted trace through the real candidate and executor paths."""
    raw = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
    fixture = _mapping(raw, "fixture")
    initial = _mapping(fixture.get("initial"), "initial")
    steps = _steps(fixture.get("steps"))
    world = _ScriptedWorld(initial)
    ledger = executor.ActionLedger()
    initial_buildings = initial.get("buildings", [])
    if not isinstance(initial_buildings, list) or not all(
        isinstance(item, str) for item in initial_buildings
    ):
        raise ValueError("initial.buildings must be a list of names")
    ledger.buildings_confirmed.update(initial_buildings)
    ledger.villagers_ordered = _integer(
        initial.get("villagers_ordered", world.population), "villagers_ordered"
    )
    source = _ScriptedSource(world, ledger)
    failures: list[str] = []
    inputs: list[str] = []
    actions: list[str] = []
    token = executor.bind_ledger(ledger)
    original = (
        executor.pyautogui,
        executor.get_game_window_rect,
        executor.ensure_game_focused,
        executor.get_rescan_fn(),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="agent-scenario-") as temporary:
            ctx = LoopContext(
                memory=AgentMemory(),
                goal_manager=GoalManager(),
                goal_logger=GoalLogger(Path(temporary)),
                source=source,
                actuator=GameActuator(),
                ledger=ledger,
            )
            ctx.memory.action_ledger = ledger
            executor.get_game_window_rect = lambda: (0, 0, 1920, 1080)
            executor.ensure_game_focused = lambda: True
            tick = 0

            async def refresh() -> bool:
                nonlocal tick
                tick += 1
                await perceive_once(ctx, tick)
                return True

            executor.set_rescan_fn(refresh)
            tick += 1
            await perceive_once(ctx, tick)
            for index, step in enumerate(steps, start=1):
                fake_input = _ScriptedInput(world, ledger, step)
                executor.pyautogui = cast("object", fake_input)
                advisor = _ScriptedAdvisor(step.action)
                frame = ctx.frames.latest()
                if frame is None:
                    raise RuntimeError("scripted source published no frame")
                await act_once(ctx, advisor, frame, index)
                actions.append(step.action)
                inputs.extend(fake_input.calls)
                if not advisor.advertised:
                    failures.append(f"step {index}: {step.action} was not eligible")
                    break
                if step.action not in {"wait", "tactical_handoff"} and not fake_input.issued:
                    failures.append(f"step {index}: no spending input issued for {step.action}")
                    break
                tick += 1
                await perceive_once(ctx, tick)
                if ledger.failure_streak:
                    failures.append(f"step {index}: {ledger.recent_failures[-1]}")
                    break
    finally:
        executor.pyautogui = original[0]
        executor.get_game_window_rect = original[1]
        executor.ensure_game_focused = original[2]
        executor._rescan_fn = original[3]
        executor.clear_detected_entities()
        executor.unbind_ledger(token)
    return ScenarioResult(path.stem, not failures, failures, actions, inputs, world.age)


def run_scenario(path: Path) -> ScenarioResult:
    return asyncio.run(run_scenario_async(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    parsed = cast("dict[str, object]", vars(parser.parse_args(argv)))
    raw_paths = parsed["fixtures"]
    if not isinstance(raw_paths, list) or not all(isinstance(path, Path) for path in raw_paths):
        raise ValueError("fixtures must be paths")
    paths = list(cast("list[Path]", raw_paths))
    if parsed["all"] is True:
        paths.extend(sorted((Path(__file__).parent / "scenarios" / "production").glob("*.yaml")))
    if not paths:
        parser.error("provide a production fixture or --all")
    results = [run_scenario(path) for path in paths]
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {', '.join(result.failures)}")
    return int(not all(result.passed for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
