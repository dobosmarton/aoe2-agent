"""AoE2-lite world simulator for multi-turn evaluation.

Models enough of the AoE2 economy to let the agent's actions meaningfully
evolve state across N turns without booting the real game:
  - Flat per-turn resource accumulation (no villager-assignment tracking)
  - Villager production queue (3-tick cooldown after queue_villager)
  - Building placement (wood cost, adds building to state)
  - Feudal Age advancement (6-tick timer, prereq check)

Fidelity is deliberately low — this is a behavioral regression harness,
not a game engine. The goal is catching stuck loops, inhibitory-memory
failures, and age-transition regressions, not simulating AoE2 exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from detection.inference.detector import DetectedEntity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_SEQUENCE = ["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]

# Flat per-turn gather increments. Represents ~4-6 villagers on each resource
# at normal AoE2 gather rates over a ~10-second turn window.
FOOD_GATHER_RATE = 20.0
WOOD_GATHER_RATE = 15.0

# Building costs (wood)
BUILDING_COSTS: dict[str, int] = {
    "q": 25,  # house
    "w": 100,  # mill
    "e": 100,  # mining camp
    "r": 100,  # lumber camp
    "a": 60,  # farm
    "s": 150,  # blacksmith
    "t": 150,  # dock
}

BUILDING_NAMES: dict[str, str] = {
    "q": "house",
    "w": "mill",
    "e": "mining_camp",
    "r": "lumber_camp",
    "a": "farm",
    "s": "blacksmith",
    "t": "dock",
}

VILLAGER_COST_FOOD = 50
VILLAGER_PRODUCTION_TICKS = 3  # turns until a queued villager is added to pop

AGE_UP_COST_FOOD = 500
AGE_UP_TICKS = 6  # turns until age advance completes

# Feudal Age prerequisites
FEUDAL_PREREQ_BUILDINGS = frozenset({"mill", "lumber_camp"})
FEUDAL_AGE_MIN_POP = 22
FEUDAL_AGE_MIN_FOOD = 500

HOUSE_POP_SLOTS = 5

# Perception-projection constants (used by render() at the bottom of this file).
_DEFAULT_RENDER_WIDTH = 1920
_DEFAULT_RENDER_HEIGHT = 1080
_TC_JITTER_X_PX = 100
_TC_JITTER_Y_PX = 50
_VILLAGER_SCATTER_RADIUS_X_PX = 150
_VILLAGER_SCATTER_RADIUS_Y_PX = 100
_BUILDING_GRID_COLS = 4
_BUILDING_GRID_SPACING_PX = 120
_BUILDING_GRID_OFFSET_PX = 220
_GROUND_TRUTH_CONFIDENCE = 1.0

# Per-class bbox dimensions (width, height) in pixels. Used by _make_entity()
# to size the projected DetectedEntity. Unit/animal values match the existing
# `detection/inference/mock.py` mock fixture; building values are sized for
# Dark/Feudal-Age footprints.
_CLASS_BBOX_DIMS_PX: dict[str, tuple[int, int]] = {
    "town_center": (160, 120),
    "villager": (24, 25),
    "house": (60, 60),
    "mill": (80, 80),
    "mining_camp": (80, 80),
    "lumber_camp": (80, 80),
    "farm": (100, 100),
    "blacksmith": (80, 80),
    "dock": (100, 80),
}


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    food: float
    wood: float
    gold: float
    stone: float
    population: int
    pop_cap: int
    age: str  # "Dark Age" | "Feudal Age" | ...
    buildings: list[str]  # may contain duplicates (e.g. multiple houses)
    villager_queue: list[int]  # countdown ticks remaining per pending villager
    age_up_ticks_remaining: int  # 0 = not in progress
    turn: int = 0


def init_from_fixture(inputs: dict) -> WorldState:
    """Build initial WorldState from a fixture `inputs` block."""
    resources = inputs.get("resources", {})
    pop_str = str(resources.get("population", "0/25"))
    pop_now_str, _, pop_cap_str = pop_str.partition("/")

    # Seed buildings from detected entity classes
    entity_classes = {e.get("class", "") for e in inputs.get("detected_entities", [])}
    known_buildings = set(BUILDING_NAMES.values())
    buildings = [cls for cls in entity_classes if cls in known_buildings]

    return WorldState(
        food=float(resources.get("food", 200)),
        wood=float(resources.get("wood", 200)),
        gold=float(resources.get("gold", 0)),
        stone=float(resources.get("stone", 0)),
        population=int(pop_now_str or 0),
        pop_cap=int(pop_cap_str or 25),
        age=inputs.get("age", "Dark Age"),
        buildings=buildings,
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def state_to_fixture_inputs(state: WorldState, base_inputs: dict) -> dict:
    """Merge world state back into the fixture `inputs` schema for the next LLM context."""
    return {
        **base_inputs,
        "age": state.age,
        "resources": {
            **base_inputs.get("resources", {}),
            "food": int(state.food),
            "wood": int(state.wood),
            "gold": int(state.gold),
            "stone": int(state.stone),
            "population": f"{state.population}/{state.pop_cap}",
        },
    }


# ---------------------------------------------------------------------------
# Action effect handlers (each returns a new WorldState)
# ---------------------------------------------------------------------------


def _apply_queue_villager(state: WorldState) -> WorldState:
    if state.food < VILLAGER_COST_FOOD:
        return state
    return replace(
        state,
        food=state.food - VILLAGER_COST_FOOD,
        villager_queue=[*state.villager_queue, VILLAGER_PRODUCTION_TICKS],
    )


def _apply_build(state: WorldState, building_key: str) -> WorldState:
    cost = BUILDING_COSTS.get(building_key, 0)
    name = BUILDING_NAMES.get(building_key)
    if not name or state.wood < cost:
        return state
    new_pop_cap = state.pop_cap + HOUSE_POP_SLOTS if name == "house" else state.pop_cap
    return replace(
        state, wood=state.wood - cost, pop_cap=new_pop_cap, buildings=[*state.buildings, name]
    )


def _feudal_prereqs_met(state: WorldState) -> bool:
    if state.age != "Dark Age":
        return False
    if state.food < FEUDAL_AGE_MIN_FOOD:
        return False
    if state.population < FEUDAL_AGE_MIN_POP:
        return False
    return FEUDAL_PREREQ_BUILDINGS.issubset(set(state.buildings))


def _apply_age_up(state: WorldState) -> WorldState:
    if state.age_up_ticks_remaining > 0:
        return state  # already in progress
    if not _feudal_prereqs_met(state):
        return state  # prereqs not met — no-op
    return replace(state, food=state.food - AGE_UP_COST_FOOD, age_up_ticks_remaining=AGE_UP_TICKS)


def apply_actions(state: WorldState, actions: list[dict]) -> WorldState:
    """Apply a list of LLM actions to the world state. Returns a new state."""
    for action in actions:
        action_type = action.get("type")
        if action_type == "queue_villager":
            state = _apply_queue_villager(state)
        elif action_type == "build":
            state = _apply_build(state, action.get("building_key", ""))
        elif action_type == "press" and action.get("key") == "z":
            state = _apply_age_up(state)
    return state


# ---------------------------------------------------------------------------
# World tick (call after apply_actions to advance one turn)
# ---------------------------------------------------------------------------


def _next_age(current_age: str) -> str:
    try:
        idx = AGE_SEQUENCE.index(current_age)
        return AGE_SEQUENCE[min(idx + 1, len(AGE_SEQUENCE) - 1)]
    except ValueError:
        return current_age


def tick(state: WorldState) -> WorldState:
    """Advance the world by one turn: gather resources, complete villagers, advance age.

    Call this AFTER apply_actions() at the end of each turn.
    """
    # Complete villagers whose countdown hit 0
    new_queue = []
    new_pop = state.population
    for countdown in state.villager_queue:
        remaining = countdown - 1
        if remaining <= 0:
            new_pop += 1
        else:
            new_queue.append(remaining)

    # Advance age-up timer
    new_age_ticks = state.age_up_ticks_remaining
    new_age = state.age
    if new_age_ticks > 0:
        new_age_ticks -= 1
        if new_age_ticks == 0:
            new_age = _next_age(state.age)

    return replace(
        state,
        food=state.food + FOOD_GATHER_RATE,
        wood=state.wood + WOOD_GATHER_RATE,
        population=new_pop,
        pop_cap=state.pop_cap,
        villager_queue=new_queue,
        age=new_age,
        age_up_ticks_remaining=new_age_ticks,
        turn=state.turn + 1,
    )


# ---------------------------------------------------------------------------
# End-state evaluation
# ---------------------------------------------------------------------------


def evaluate_end_state(end_state_spec: dict, state: WorldState) -> list[str]:
    """Check end-state assertions against the final WorldState.

    Numeric fields use ≥ semantics (e.g. population: 15 → at least 15).
    String fields use exact equality (e.g. age: "Feudal Age").
    """
    failures = []
    for key, expected in end_state_spec.items():
        actual = getattr(state, key, None)
        if actual is None:
            failures.append(f"end_state: unknown WorldState field {key!r}")
            continue
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            if actual < expected:
                failures.append(
                    f"end_state FAILED: {key}={actual} < expected ≥ {expected} "
                    f"(after {state.turn} turns)"
                )
        elif actual != expected:
            failures.append(
                f"end_state FAILED: {key}={actual!r} != expected {expected!r} "
                f"(after {state.turn} turns)"
            )
    return failures


# ---------------------------------------------------------------------------
# Rendering: project WorldState to DetectedEntity[] for synthetic perception.
#
# Phase 1 of the synthetic-arena buildout — see
# docs/design/synthetic-arena-analysis.md lines 336 (deliverable) and 380
# (Risk 4: schema contract).
# ---------------------------------------------------------------------------


def _make_entity(
    class_name: str,
    center: tuple[float, float],
    id_factory: Callable[[str], str],
) -> DetectedEntity:
    from detection.inference.detector import DetectedEntity

    bbox_width, bbox_height = _CLASS_BBOX_DIMS_PX[class_name]
    center_x, center_y = center
    half_width = bbox_width / 2
    half_height = bbox_height / 2
    return DetectedEntity(
        id=id_factory(class_name),
        class_name=class_name,
        bbox=(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        center=center,
        confidence=_GROUND_TRUTH_CONFIDENCE,
        area=float(bbox_width * bbox_height),
    )


def _render_town_center(
    rng: random.Random,
    width: int,
    height: int,
    id_factory: Callable[[str], str],
) -> DetectedEntity:
    center_x = width * 0.5 + rng.uniform(-_TC_JITTER_X_PX, _TC_JITTER_X_PX)
    center_y = height * 0.5 + rng.uniform(-_TC_JITTER_Y_PX, _TC_JITTER_Y_PX)
    return _make_entity("town_center", (center_x, center_y), id_factory)


def _render_villagers(
    state: WorldState,
    rng: random.Random,
    tc_center: tuple[float, float],
    id_factory: Callable[[str], str],
) -> list[DetectedEntity]:
    # villager_queue entries are deliberately excluded: they are queued for
    # production at the TC and not yet on the map for the agent to perceive.
    tc_x, tc_y = tc_center
    villagers: list[DetectedEntity] = []
    for _ in range(state.population):
        vill_x = tc_x + rng.uniform(-_VILLAGER_SCATTER_RADIUS_X_PX, _VILLAGER_SCATTER_RADIUS_X_PX)
        vill_y = tc_y + rng.uniform(-_VILLAGER_SCATTER_RADIUS_Y_PX, _VILLAGER_SCATTER_RADIUS_Y_PX)
        villagers.append(_make_entity("villager", (vill_x, vill_y), id_factory))
    return villagers


def _render_buildings(
    state: WorldState,
    tc_center: tuple[float, float],
    id_factory: Callable[[str], str],
) -> list[DetectedEntity]:
    # Index-based grid keeps positions stable when buildings are appended.
    tc_x, tc_y = tc_center
    entities: list[DetectedEntity] = []
    for index, name in enumerate(state.buildings):
        col = index % _BUILDING_GRID_COLS
        row = index // _BUILDING_GRID_COLS
        building_x = tc_x + _BUILDING_GRID_OFFSET_PX + col * _BUILDING_GRID_SPACING_PX
        building_y = tc_y + row * _BUILDING_GRID_SPACING_PX
        entities.append(_make_entity(name, (building_x, building_y), id_factory))
    return entities


def _default_id_factory() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def make_id(class_name: str) -> str:
        index = counters.get(class_name, 0)
        counters[class_name] = index + 1
        return f"{class_name}_{index}"

    return make_id


def render(
    state: WorldState,
    width: int = _DEFAULT_RENDER_WIDTH,
    height: int = _DEFAULT_RENDER_HEIGHT,
    id_factory: Callable[[str], str] | None = None,
    seed: int | None = None,
) -> list[DetectedEntity]:
    """Project a WorldState to a list of DetectedEntity at the given dimensions.

    Same state + same dims (+ same seed) ⇒ identical output. Uses a local
    random.Random instance — never mutates global random state.
    """
    factory = id_factory if id_factory is not None else _default_id_factory()
    rng = random.Random(seed if seed is not None else state.turn)

    town_center = _render_town_center(rng, width, height, factory)
    villagers = _render_villagers(state, rng, town_center.center, factory)
    buildings = _render_buildings(state, town_center.center, factory)

    entities = [town_center, *villagers, *buildings]
    entities.sort(key=lambda e: (e.class_name, -e.confidence))
    return entities
