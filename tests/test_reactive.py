"""Unit tests for S5 deterministic reactive tier (gameplay_agent/reactive.py).

Pure functions over fake entity dicts + GameState. No executor / pyautogui.
"""

from __future__ import annotations

import pytest
from gameplay_agent.entity_utils import nearest_class_of_kind
from gameplay_agent.memory import GameState
from gameplay_agent.reactive import _resolve_idle_target, decide

from tests.factories import make_entity as _ent


def _state(
    population: int,
    population_cap: int = 30,
    age: str = "Dark Age",
    idle_present: bool | None = None,
    idle_count: int | None = None,
    idle_streak: int = 0,
    food: int = 200,
    wood: int = 200,
    # Default = Feudal prereqs satisfied, so tests about OTHER features aren't
    # polluted by the lumber-camp prep action; prep/age-up tests override it.
    buildings: frozenset[str] = frozenset({"mill", "lumber_camp"}),
) -> GameState:
    return GameState(
        resources={"food": food, "wood": wood, "gold": 100, "stone": 200},
        population=population,
        population_cap=population_cap,
        current_age=age,
        idle_present=idle_present,
        idle_count=idle_count,
        idle_streak=idle_streak,
        buildings_seen=buildings,
    )


def _types(actions: list[dict]) -> list[str]:
    return [str(a.get("type")) for a in actions]


# ---------------------------------------------------------------------------
# Villager queuing
# ---------------------------------------------------------------------------


def test_queues_villager_below_cap() -> None:
    actions = decide([], _state(population=10), alarm=False)
    assert _types(actions) == ["press", "press"]
    assert actions[1] == {"type": "press", "key": "q", "intent": "Queue villager (reactive)"}


def test_no_queue_at_dark_age_cap() -> None:
    # Dark Age cap is 22; pop 22 is not below it → no queue.
    actions = decide([], _state(population=22), alarm=False)
    assert actions == []


# ---------------------------------------------------------------------------
# Feudal banking + age-up (T-510 / T-511)
# ---------------------------------------------------------------------------


def test_age_up_fires_when_food_banked_and_buildings_qualify() -> None:
    actions = decide([], _state(population=18, food=520), alarm=False)
    assert actions == [
        {"type": "press", "key": "escape", "intent": "Clear UI state (age up)"},
        {"type": "press", "key": "h", "intent": "Select TC (age up)"},
        {"type": "press", "key": "z", "intent": "Research Feudal Age (reactive)"},
    ]  # banking (pop 18): no villager queue rides along to eat the 500;
    # escape first so `z` can't land in an open build menu (F-27's outposts)


def test_age_up_waits_for_two_qualifying_buildings() -> None:
    """Run 6 (F-26): 500+ food banked, only the mill built — 14 presses no-oped
    against a greyed button. Now the press waits and the prep builds the camp."""
    state = _state(population=18, food=520, buildings=frozenset({"mill"}))
    actions = decide([], state, alarm=False)
    assert all(a.get("key") != "z" for a in actions)  # no futile age-up press
    assert [a.get("building_key") for a in actions if a["type"] == "build"] == ["r"]


def test_feudal_prep_builds_lumber_camp_once_economy_established() -> None:
    state = _state(population=12, buildings=frozenset({"mill"}))
    actions = decide([], state, alarm=False)
    builds = [a for a in actions if a["type"] == "build"]
    assert builds == [
        {
            "type": "build",
            "building_key": "r",
            "intent": "Build lumber camp (Feudal prerequisite + wood income)",
        }
    ]


@pytest.mark.parametrize(
    ("population", "age", "buildings"),
    [
        (11, "Dark Age", frozenset({"mill"})),  # economy not established yet
        (12, "Dark Age", frozenset({"mill", "lumber_camp"})),  # already standing
        (12, "Feudal Age", frozenset({"mill"})),  # prereq is a Dark Age concern
    ],
    ids=["below-prep-pop", "camp-exists", "wrong-age"],
)
def test_feudal_prep_not_emitted(population: int, age: str, buildings: frozenset[str]) -> None:
    actions = decide([], _state(population=population, age=age, buildings=buildings), alarm=False)
    assert all(a.get("type") != "build" for a in actions)


def test_feudal_prereq_classes_match_world_sim() -> None:
    """Drift guard (V-4): run 6 proved the sim knew the requirement the agent
    lacked — keep the two encodings pinned together."""
    from evaluation.world_sim import FEUDAL_PREREQ_BUILDINGS
    from gameplay_agent import reactive

    assert reactive._FEUDAL_PREREQ_CLASSES == FEUDAL_PREREQ_BUILDINGS


@pytest.mark.parametrize(
    ("age", "food"),
    [("Dark Age", 499), ("Feudal Age", 600)],
    ids=["below-cost", "outside-dark-age"],
)
def test_no_age_up_when_preconditions_missing(age: str, food: int) -> None:
    actions = decide([], _state(population=18, age=age, food=food), alarm=False)
    assert all(a.get("key") != "z" for a in actions)


@pytest.mark.parametrize(
    ("population", "age", "queues"),
    [
        (16, "Dark Age", False),  # established economy → bank for Feudal
        (15, "Dark Age", True),  # still growing → keep queueing
        (16, "Feudal Age", True),  # banking is Dark Age only (Feudal cap is 35)
    ],
    ids=["dark-age-banks", "below-threshold-queues", "feudal-queues"],
)
def test_villager_queue_respects_feudal_banking(population: int, age: str, queues: bool) -> None:
    actions = decide([], _state(population=population, age=age), alarm=False)
    assert (_types(actions) == ["press", "press"]) is queues


def test_food_crisis_forces_all_idle_slots_to_food() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    # pop 22 phases would be food, wood, wood — the crisis override sends all to food.
    actions = decide(entities, _state(population=22, idle_present=True, food=40), alarm=False)
    targets = [a["target_class"] for a in actions if a["type"] == "right_click"]
    assert targets == ["sheep"] * 3


def test_food_crisis_without_forage_builds_one_farm() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("tree", (20, 20))]
    actions = decide(entities, _state(population=22, idle_present=True, food=40), alarm=False)
    assert [a["building_key"] for a in actions if a["type"] == "build"] == ["a"]


def test_food_crisis_with_no_wood_reserves_a_wood_slot() -> None:
    """Run 4 (F-21): the pure all-food famine override pinned wood at 0 and
    locked out the farm economy. With wood below a farm's cost, the override
    routes 2:1 food:wood instead."""
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    actions = decide(
        entities, _state(population=22, idle_present=True, food=40, wood=0), alarm=False
    )
    targets = sorted(a["target_class"] for a in actions if a["type"] == "right_click")
    assert targets == ["sheep", "sheep", "tree"]  # exactly one wood slot per 3-batch


def test_farm_wood_cost_matches_executor_table() -> None:
    """Drift guard (V-4 seed): the reactive tier duplicates the farm cost to
    stay dependency-free — this pins the two copies together."""
    from gameplay_agent import executor as ex
    from gameplay_agent import reactive

    assert ex._BUILD_WOOD_COST["a"] == reactive._FARM_WOOD_COST


# ---------------------------------------------------------------------------
# Safe huntables (T-515): boar/deer are never gather targets
# ---------------------------------------------------------------------------


def test_boar_never_dispatched_even_when_only_food_nearby() -> None:
    """Run 4 (F-20): 9 boar dispatches killed 3 villagers — a lone right-click
    on a boar is an attack. A visible boar must trigger the farm-build path,
    never a dispatch."""
    entities = [_ent("town_center", (0, 0)), _ent("boar", (10, 10))]
    actions = decide(entities, _state(population=22, idle_present=True), alarm=False)
    assert all(a.get("target_class") != "boar" for a in actions)
    assert [a["building_key"] for a in actions if a["type"] == "build"] == ["a"]


def test_sheep_preferred_boar_and_deer_ignored() -> None:
    entities = [
        _ent("town_center", (0, 0)),
        _ent("boar", (5, 5)),  # nearest, but lethal
        _ent("deer", (8, 8)),  # unsafe to trust at F1 0.67
        _ent("sheep", (50, 50)),
    ]
    actions = decide(entities, _state(population=20, idle_present=True), alarm=False)
    targets = {a["target_class"] for a in actions if a["type"] == "right_click"}
    assert targets == {"sheep"}


# ---------------------------------------------------------------------------
# Idle-villager distribution (gated on the HUD badge count)
# ---------------------------------------------------------------------------


def test_idle_dispatches_single_dot_not_blanket() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # pop 22 = at Dark-Age cap (no queue); badge present → a batch of single-`.` sends.
    actions = decide(entities, _state(population=22, idle_present=True), alarm=False)
    assert _types(actions) == ["press", "right_click"] * 3  # _IDLE_DISPATCH_PER_TURN
    press = actions[0]
    assert press["key"] == "." and "modifiers" not in press  # single idle, not Shift-.
    assert press.get("rescan") is True
    assert actions[1]["target_class"] == "sheep"  # only food visible


def test_no_dispatch_when_badge_absent_or_unknown() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # False = none idle, None = badge unread — both must skip (no wasted camera move).
    assert decide(entities, _state(population=22, idle_present=False), alarm=False) == []
    assert decide(entities, _state(population=22, idle_present=None), alarm=False) == []


def test_idle_food_turn_without_food_builds_farm() -> None:
    # pop 22 → phases food, wood, wood. Nothing gatherable on screen: the food
    # slot builds a fresh farm (executor gates decide if it can actually work);
    # the wood slots have no target and stop the batch.
    actions = decide([_ent("villager")], _state(population=22, idle_present=True), alarm=False)
    assert actions == [
        {
            "type": "build",
            "building_key": "a",
            "intent": "Build farm for idle villager (no forage/huntables visible)",
        }
    ]


def test_idle_never_targets_farms_builds_one_instead() -> None:
    # Farms are not gather targets (misdetected bare ground strands villagers;
    # occupied farms take one villager each) — a visible farm changes nothing.
    entities = [_ent("town_center", (0, 0)), _ent("farm", (10, 10))]
    actions = decide(entities, _state(population=22, idle_present=True), alarm=False)
    builds = [a for a in actions if a["type"] == "build"]
    assert len(builds) == 1 and builds[0]["building_key"] == "a"
    assert all(a.get("target_class") != "farm" for a in actions)


def test_idle_food_turn_prefers_huntables_over_farm_build() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("farm", (20, 20))]
    actions = decide(entities, _state(population=20, idle_present=True), alarm=False)
    # pop 20 → all-food phases: sheep is targeted, no farm build, farm untouched.
    targets = [a["target_class"] for a in actions if a["type"] == "right_click"]
    assert targets == ["sheep"] * 3
    assert all(a["type"] != "build" for a in actions)


def test_idle_farm_build_capped_at_one_per_turn() -> None:
    # pop 20 → Feudal banking (no queue) and all three idle phases are food,
    # none gatherable → exactly ONE farm build (this turn's HUD snapshot can't
    # cost-check a second spend).
    actions = decide([_ent("town_center")], _state(population=20, idle_present=True), alarm=False)
    assert [a["type"] for a in actions] == ["build"]


def test_idle_dispatch_capped_per_turn() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    actions = decide(entities, _state(population=22, idle_present=True), alarm=False)
    # Presence only (no count) → a fixed _IDLE_DISPATCH_PER_TURN (3) batch.
    assert _types(actions) == ["press", "right_click"] * 3


def test_idle_batch_sized_by_badge_count() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # Known count 2 → exactly 2 dispatches (not the blind 3).
    actions = decide(entities, _state(population=22, idle_present=True, idle_count=2), alarm=False)
    assert _types(actions) == ["press", "right_click"] * 2


def test_idle_batch_capped_on_mass_idle() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # Count 18 (post-combat pile-up) → capped at _IDLE_DISPATCH_MAX (6) per turn.
    actions = decide(entities, _state(population=22, idle_present=True, idle_count=18), alarm=False)
    assert _types(actions) == ["press", "right_click"] * 6


def test_idle_count_trusted_below_streak_threshold() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # Badge lit 3 turns (below the suspect threshold): the digit is still trusted.
    state = _state(population=22, idle_present=True, idle_count=1, idle_streak=3)
    assert _types(decide(entities, state, alarm=False)) == ["press", "right_click"] * 1


def test_idle_count_distrusted_on_long_streak() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # Badge lit 4+ turns while the digit reads 1 — the 2026-07-11 failure (digit
    # pinned at 1 with 8 idle). Floor the batch at the blind presence default.
    state = _state(population=22, idle_present=True, idle_count=1, idle_streak=4)
    assert _types(decide(entities, state, alarm=False)) == ["press", "right_click"] * 3


def test_idle_count_zero_distrusted_on_long_streak() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # A pinned 0 with a lit badge is the same sensor fault as a pinned 1.
    state = _state(population=22, idle_present=True, idle_count=0, idle_streak=4)
    assert _types(decide(entities, state, alarm=False)) == ["press", "right_click"] * 3


def test_idle_count_gate_never_reduces_batch() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # A big count on a long streak keeps its exact size (and the per-turn cap).
    state = _state(population=22, idle_present=True, idle_count=5, idle_streak=9)
    assert _types(decide(entities, state, alarm=False)) == ["press", "right_click"] * 5
    capped = _state(population=22, idle_present=True, idle_count=18, idle_streak=9)
    assert _types(decide(entities, capped, alarm=False)) == ["press", "right_click"] * 6


def test_idle_count_zero_overrides_presence() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    # Badge colour said present but the digit reads 0 → trust the digit, skip.
    actions = decide(entities, _state(population=22, idle_present=True, idle_count=0), alarm=False)
    assert actions == []


def test_idle_distribution_spreads_across_kinds() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    # Dark-Age pattern ("food","food","food","wood","wood"); phase = population + i.
    # pop 22 → phases 2,3,4 → food, wood, wood → sheep, tree, tree (spread in ONE turn).
    turn = decide(entities, _state(population=22, idle_present=True), alarm=False)
    targets = [a["target_class"] for a in turn if a["type"] == "right_click"]
    assert targets == ["sheep", "tree", "tree"]
    # pop 20 → phases 0,1,2 → all food → all sheep (rotation shifts by population).
    turn2 = decide(entities, _state(population=20, idle_present=True), alarm=False)
    assert [a["target_class"] for a in turn2 if a["type"] == "right_click"] == ["sheep"] * 3


# ---------------------------------------------------------------------------
# Resource targeting helpers
# ---------------------------------------------------------------------------


def test_nearest_class_of_kind_picks_nearest_to_origin() -> None:
    entities = [_ent("sheep", (10, 10)), _ent("boar", (900, 900))]
    assert nearest_class_of_kind(entities, "food", origin=(0, 0)) == "sheep"


def test_nearest_class_of_kind_none_when_absent() -> None:
    assert nearest_class_of_kind([_ent("villager")], "food") is None


def test_resolve_idle_target_falls_through_priority() -> None:
    # Requested gold, but only food is visible → fall through to the visible kind.
    entities = [_ent("sheep", (10, 10))]
    assert _resolve_idle_target(entities, "gold", origin=(0, 0)) == "sheep"
    # Nothing gatherable → None.
    assert _resolve_idle_target([_ent("villager")], "food", origin=(0, 0)) is None


# ---------------------------------------------------------------------------
# Alarm + determinism
# ---------------------------------------------------------------------------


def test_returns_empty_on_alarm() -> None:
    entities = [_ent("sheep", (10, 10))]
    assert decide(entities, _state(population=5, idle_present=True), alarm=True) == []


def test_decide_is_deterministic() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    state = _state(population=10, idle_present=True)
    assert decide(entities, state, alarm=False) == decide(entities, state, alarm=False)
