"""Unit tests for S5 deterministic reactive tier (gameplay_agent/reactive.py).

Pure functions over fake entity dicts + GameState. No executor / pyautogui.
"""

from __future__ import annotations

import pytest
from gameplay_agent.entity_utils import nearest_class_of_kind
from gameplay_agent.memory import GameState
from gameplay_agent.reactive import _idle_pattern, _resolve_idle_target, decide

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
    gold: int = 100,
    # Default = every prep goal satisfied (Feudal prereqs, gold drop-off, and
    # the two Castle prereqs), so tests about OTHER features aren't polluted by
    # prep build actions; prep/age-up tests override it.
    buildings: frozenset[str] = frozenset(
        {"mill", "lumber_camp", "mining_camp", "blacksmith", "market"}
    ),
    # Default = villager target reached, so no queue order rides along in
    # dispatch/age-up tests; queue tests pass explicit lower orders.
    villagers_ordered: int = 30,
) -> GameState:
    return GameState(
        resources={"food": food, "wood": wood, "gold": gold, "stone": 200},
        population=population,
        population_cap=population_cap,
        current_age=age,
        idle_present=idle_present,
        idle_count=idle_count,
        idle_streak=idle_streak,
        buildings_seen=buildings,
        villagers_ordered=villagers_ordered,
    )


def _types(actions: list[dict]) -> list[str]:
    return [str(a.get("type")) for a in actions]


# ---------------------------------------------------------------------------
# Villager queuing
# ---------------------------------------------------------------------------


def test_queues_villager_below_target() -> None:
    actions = decide([], _state(population=10, villagers_ordered=10), alarm=False)
    assert actions == [{"type": "queue_villager", "intent": "Queue villager (reactive)"}]


def test_no_queue_at_dark_age_order_target() -> None:
    # 30 villagers ORDERED (user directive, F-38) → bank food, don't queue —
    # regardless of how few the TC queue has delivered yet.
    actions = decide([], _state(population=22, villagers_ordered=30), alarm=False)
    assert actions == []


# ---------------------------------------------------------------------------
# Feudal banking + age-up (T-510 / T-511)
# ---------------------------------------------------------------------------


def test_age_up_fires_when_food_banked_and_buildings_qualify() -> None:
    actions = decide([], _state(population=18, food=520), alarm=False)
    assert actions == [
        {"type": "press", "key": "h", "intent": "Select TC (age up)"},
        {"type": "press", "key": "z", "intent": "Research Feudal Age (reactive)"},
    ]  # banking (pop 18): no villager queue rides along to eat the 500;
    # selecting the TC first clears any open build menu so `z` can't land in
    # it (F-27's outposts) — and never opens the game menu (F-32)


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


def test_feudal_prep_builds_mill_first_when_neither_stands() -> None:
    """Run 12 (F-41): with the executor down, the reactive tier had no mill
    path and starved. The mill (farm unlock) must lead when neither prereq
    stands — one build per turn, so the camp waits its turn."""
    state = _state(population=12, buildings=frozenset())
    builds = [a for a in decide([], state, alarm=False) if a["type"] == "build"]
    assert builds == [
        {
            "type": "build",
            "building_key": "w",
            "intent": "Build mill (Feudal prerequisite + farm/food unlock)",
        }
    ]


def test_feudal_prep_builds_camp_after_mill_stands() -> None:
    """Once the mill is confirmed, prep moves on to the lumber camp."""
    state = _state(population=12, buildings=frozenset({"mill"}))
    keys = [a.get("building_key") for a in decide([], state, alarm=False) if a["type"] == "build"]
    assert keys == ["r"]  # mill satisfied → camp next, never a second mill


@pytest.mark.parametrize(
    ("population", "age", "buildings"),
    [
        (11, "Dark Age", frozenset({"mill"})),  # economy not established yet
        (12, "Dark Age", frozenset({"mill", "lumber_camp"})),  # already standing
        # Prereqs are a Dark Age concern (the Castle prep's own goals are all
        # satisfied here so it doesn't emit a build of its own).
        (12, "Feudal Age", frozenset({"mill", "mining_camp", "blacksmith", "market"})),
    ],
    ids=["below-prep-pop", "camp-exists", "wrong-age"],
)
def test_feudal_prep_not_emitted(population: int, age: str, buildings: frozenset[str]) -> None:
    actions = decide([], _state(population=population, age=age, buildings=buildings), alarm=False)
    assert all(a.get("type") != "build" for a in actions)


def test_castle_prep_builds_mining_camp_in_feudal() -> None:
    """T-538 (run 13, F-46): 25 minutes in Feudal with gold parked at 90 —
    no tier ever built the gold drop-off. The reactive tier now does."""
    state = _state(population=25, age="Feudal Age", buildings=frozenset({"mill", "lumber_camp"}))
    builds = [a for a in decide([], state, alarm=False) if a["type"] == "build"]
    assert builds == [
        {
            "type": "build",
            "building_key": "e",
            "intent": "Build mining camp (gold drop-off for the Castle Age bank)",
        }
    ]


@pytest.mark.parametrize(
    ("age", "buildings"),
    [
        ("Dark Age", frozenset({"mill", "lumber_camp"})),  # gold is a Feudal concern
        ("Feudal Age", frozenset({"mill", "lumber_camp", "mining_camp"})),  # standing
    ],
    ids=["wrong-age", "camp-exists"],
)
def test_castle_prep_not_emitted(age: str, buildings: frozenset[str]) -> None:
    actions = decide([], _state(population=25, age=age, buildings=buildings), alarm=False)
    assert all(a.get("building_key") != "e" for a in actions)


def test_castle_prep_builds_blacksmith_once_the_gold_engine_stands() -> None:
    """T-544: with the drop-off up, prep moves to the two-building requirement —
    blacksmith first (cheaper, and on the verified econ menu)."""
    state = _state(
        population=25,
        age="Feudal Age",
        buildings=frozenset({"mill", "lumber_camp", "mining_camp"}),
    )
    builds = [a for a in decide([], state, alarm=False) if a["type"] == "build"]
    assert builds == [
        {
            "type": "build",
            "building_key": "s",
            "intent": "Build blacksmith (Castle Age prerequisite 1 of 2)",
        }
    ]


def test_castle_prep_builds_market_last() -> None:
    """The second qualifying building, on the more-buildings menu (`vd`). One
    build per turn, so it only comes up once the blacksmith stands."""
    state = _state(
        population=25,
        age="Feudal Age",
        buildings=frozenset({"mill", "lumber_camp", "mining_camp", "blacksmith"}),
    )
    builds = [a for a in decide([], state, alarm=False) if a["type"] == "build"]
    assert builds == [
        {
            "type": "build",
            "building_key": "vd",
            "intent": "Build market (Castle Age prerequisite 2 of 2)",
        }
    ]


def test_castle_age_up_fires_when_food_gold_and_both_buildings_are_ready() -> None:
    """Run 13 reached Feudal and stopped there. The press needs 800 food, 200
    gold AND two Feudal-age buildings — the Dark Age lesson (F-26) one age up."""
    state = _state(
        population=32,
        age="Feudal Age",
        food=820,
        gold=210,
        buildings=frozenset({"mill", "lumber_camp", "mining_camp", "blacksmith", "market"}),
        villagers_ordered=35,
    )
    actions = decide([], state, alarm=False)
    assert actions[:2] == [
        {"type": "press", "key": "h", "intent": "Select TC (age up)"},
        {"type": "press", "key": "z", "intent": "Research Castle Age (reactive)"},
    ]


@pytest.mark.parametrize(
    ("food", "gold", "buildings"),
    [
        (799, 210, frozenset({"blacksmith", "market"})),
        (820, 199, frozenset({"blacksmith", "market"})),
        (820, 210, frozenset({"blacksmith"})),
    ],
    ids=["food-short", "gold-short", "one-building-short"],
)
def test_no_castle_age_up_when_preconditions_missing(
    food: int, gold: int, buildings: frozenset[str]
) -> None:
    """A press against a greyed button is not free — run 6 spent 14 of them,
    each a chance to leak UI context into the next keystroke (F-27)."""
    state = _state(
        population=32,
        age="Feudal Age",
        food=food,
        gold=gold,
        buildings=buildings | {"mill", "lumber_camp", "mining_camp"},
        villagers_ordered=35,
    )
    assert all(a.get("key") != "z" for a in decide([], state, alarm=False))


def test_no_reactive_age_up_past_castle() -> None:
    """No Castle-age program exists yet, so the press stays the LLM's call."""
    state = _state(population=40, age="Castle Age", food=2000, gold=2000, villagers_ordered=99)
    assert all(a.get("key") != "z" for a in decide([], state, alarm=False))


def test_house_built_when_headroom_runs_out() -> None:
    """T-538: housed IS a blocked villager queue (run 13 stalled at 5/5 and
    10/10) — the fast tier emits its own house instead of waiting on the LLM."""
    state = _state(population=28, population_cap=30, villagers_ordered=28)
    actions = decide([], state, alarm=False)
    keys = [a.get("building_key") for a in actions if a["type"] == "build"]
    assert keys == ["q"]


@pytest.mark.parametrize(
    ("population", "population_cap"),
    [
        (27, 30),  # headroom 3 — above the trigger, the LLM's call
        (199, 200),  # headroom 1 but cap at the game maximum: houses add nothing
    ],
    ids=["ample-headroom", "game-pop-cap"],
)
def test_house_not_built(population: int, population_cap: int) -> None:
    state = _state(
        population=population, population_cap=population_cap, villagers_ordered=population
    )
    actions = decide([], state, alarm=False)
    assert all(a.get("building_key") != "q" for a in actions)


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
    ("ordered", "age", "queues"),
    [
        (30, "Dark Age", False),  # target ordered → bank for Feudal
        (29, "Dark Age", True),  # still growing → keep ordering
        (30, "Feudal Age", True),  # Feudal target is 35 (pop cap 50 here)
        (35, "Feudal Age", False),
    ],
    ids=["dark-age-banks", "below-target-orders", "feudal-orders", "feudal-target"],
)
def test_villager_queue_respects_order_target(ordered: int, age: str, queues: bool) -> None:
    state = _state(population=15, population_cap=50, age=age, villagers_ordered=ordered)
    actions = decide([], state, alarm=False)
    assert (_types(actions) == ["queue_villager"]) is queues


def test_orders_capped_by_population_cap() -> None:
    # Housed: orders never outrun the current housing cap — and the house rule
    # (T-538) emits the un-stall build instead of leaving the turn empty.
    state = _state(population=10, population_cap=10, villagers_ordered=10)
    actions = decide([], state, alarm=False)
    assert "queue_villager" not in _types(actions)
    assert [a.get("building_key") for a in actions if a["type"] == "build"] == ["q"]


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


def test_food_crisis_wood_floor_includes_margin() -> None:
    # Famine wood routing banks a farm's cost PLUS margin: at exactly the cost
    # (run 5, F-23: six attempts failed at 48-59 wood) farms lose the race
    # against the next purchase.
    assert _idle_pattern(_state(population=22, food=40, wood=65)) == ("food", "food", "wood")
    assert _idle_pattern(_state(population=22, food=40, wood=85)) == ("food",)


# ---------------------------------------------------------------------------
# Goal-driven wood bank target (T-518 → T-527)
# ---------------------------------------------------------------------------


def test_wood_below_farm_target_gets_extra_wood_slot() -> None:
    pattern = _idle_pattern(_state(population=20, wood=50))
    assert pattern[0] == "wood"
    assert pattern.count("wood") == 3  # vs 2 in the plain Dark Age rotation


def test_lumber_camp_goal_raises_the_wood_target() -> None:
    """Run 8 (F-34): wood plateaued at 65 while the camp cost 100 — the bias
    must keep pulling wood until the CAMP is affordable, not just a farm."""
    needs_camp = _state(population=20, wood=110, buildings=frozenset({"mill"}))
    assert _idle_pattern(needs_camp)[0] == "wood"  # 110 < 100 + margin
    banked = _state(population=20, wood=125, buildings=frozenset({"mill"}))
    assert _idle_pattern(banked) == ("food", "food", "food", "wood", "wood")


@pytest.mark.parametrize(
    ("population", "wood", "buildings"),
    [
        # Below prep pop: no Feudal-prereq wood goal has opened yet.
        (11, 50, frozenset()),
        # Both prereqs standing and wood comfortably covers a farm.
        (20, 85, frozenset({"mill", "lumber_camp"})),
    ],
    ids=["below-prep-pop", "farm-banked"],
)
def test_no_wood_bias_without_pending_wood_goal(
    population: int, wood: int, buildings: frozenset[str]
) -> None:
    state = _state(population=population, wood=wood, buildings=buildings)
    assert _idle_pattern(state) == ("food", "food", "food", "wood", "wood")


def test_missing_mill_raises_the_wood_target() -> None:
    """The mill is a Feudal prereq AND the farm unlock, so once the economy is
    established a missing mill banks wood toward it — like the camp (F-41)."""
    needs_mill = _state(population=20, wood=110, buildings=frozenset())
    assert _idle_pattern(needs_mill)[0] == "wood"  # 110 < 100 + margin
    banked = _state(population=20, wood=125, buildings=frozenset())
    assert _idle_pattern(banked) == ("food", "food", "food", "wood", "wood")


def test_missing_mining_camp_raises_the_wood_target() -> None:
    """T-538: in Feudal the camp (gold drop-off) outranks the farm band, same
    arrangement as the Dark Age prereqs."""
    needs_camp = _state(
        population=25, age="Feudal Age", wood=110, buildings=frozenset({"mill", "lumber_camp"})
    )
    assert _idle_pattern(needs_camp)[0] == "wood"  # 110 < 100 + margin


@pytest.mark.parametrize(
    ("wood", "buildings"),
    [
        (160, frozenset({"mining_camp"})),  # blacksmith next: 160 < 150 + margin
        (190, frozenset({"mining_camp", "blacksmith"})),  # market: 190 < 175 + margin
    ],
    ids=["blacksmith", "market"],
)
def test_castle_prereq_buildings_raise_the_wood_target(
    wood: int, buildings: frozenset[str]
) -> None:
    """T-544: the two qualifying buildings cost 150 and 175 — the farm band (60)
    would let wood plateau below both, exactly the F-34 failure."""
    state = _state(
        population=30,
        age="Feudal Age",
        wood=wood,
        buildings=buildings | {"mill", "lumber_camp"},
    )
    assert _idle_pattern(state)[0] == "wood"


def test_feudal_gold_bias_until_castle_gold_banked() -> None:
    """T-538: below 200 gold in Feudal the rotation leads with gold; once the
    Castle gold is banked the base pattern returns. Dark Age never biases."""
    poor = _state(population=25, age="Feudal Age", wood=300)  # factory gold=100
    assert _idle_pattern(poor)[0] == "gold"
    rich = _state(population=25, age="Feudal Age", wood=300)
    rich.resources["gold"] = 250
    assert _idle_pattern(rich) == ("food", "food", "wood", "wood", "gold")
    assert _idle_pattern(_state(population=20, wood=300))[0] != "gold"  # Dark Age


def test_wood_bias_outranks_gold_bias() -> None:
    # The mining camp the gold bank needs costs wood — buildings come first.
    state = _state(
        population=25, age="Feudal Age", wood=50, buildings=frozenset({"mill", "lumber_camp"})
    )
    assert _idle_pattern(state)[0] == "wood"


def test_wood_bank_bias_sends_an_idle_to_wood() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    # pop 20 with plain rotation is all-food (see spread test below); the
    # bank bias must route at least one dispatch to wood.
    actions = decide(entities, _state(population=20, idle_present=True, wood=50), alarm=False)
    targets = [a["target_class"] for a in actions if a["type"] == "right_click"]
    assert "tree" in targets


def _gamestate_from_world(w: object) -> GameState:
    """Project a WorldState into the GameState slice reactive.decide reads.

    Idle dispatch and the villager queue are suppressed (idle_present=False,
    villagers_ordered high) so the closed loops below isolate the prep BUILD
    and age-up paths — the concern F-41 and F-46 are about.
    """
    return GameState(
        resources={
            "food": int(w.food),  # type: ignore[attr-defined]
            "wood": int(w.wood),  # type: ignore[attr-defined]
            "gold": int(w.gold),  # type: ignore[attr-defined]
            "stone": int(w.stone),  # type: ignore[attr-defined]
        },
        population=w.population,  # type: ignore[attr-defined]
        population_cap=w.pop_cap,  # type: ignore[attr-defined]
        current_age=w.age,  # type: ignore[attr-defined]
        idle_present=False,
        idle_count=None,
        buildings_seen=frozenset(w.buildings),  # type: ignore[attr-defined]
        villagers_ordered=99,
    )


def test_reactive_tier_alone_builds_both_feudal_prereqs() -> None:
    """F-41 closed loop: with NO LLM in the loop, the reactive tier must reach
    BOTH a mill and a lumber camp standing. Run 12 played 85/95 turns with the
    executor down and never built a mill (only the LLM ever did), so the food
    engine — farms, which need the mill — never started. This drives
    reactive.decide against the world_sim economy to prove the fast tier now
    gets there on its own."""
    from core import WorldState
    from evaluation.world_sim import apply_actions, tick

    state = WorldState(
        food=200.0,
        wood=100.0,
        gold=100.0,
        stone=200.0,
        population=12,  # economy established → prep gate open
        pop_cap=30,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
    )
    for _ in range(15):
        actions = decide([], _gamestate_from_world(state), alarm=False)
        state = tick(apply_actions(state, actions))
        if {"mill", "lumber_camp"}.issubset(set(state.buildings)):
            break
    assert "mill" in state.buildings, f"no mill after {state.turn} turns: {state.buildings}"
    assert "lumber_camp" in state.buildings
    assert state.buildings.count("mill") == 1  # prep never double-builds a prereq


def test_reactive_tier_alone_builds_mining_camp_in_feudal() -> None:
    """T-538 closed loop: from a fresh Feudal state, the reactive tier alone
    must stand up the gold drop-off (run 13, F-46: it never happened) — and
    stop at one (absence-gated on buildings_seen)."""
    from core import WorldState
    from evaluation.world_sim import apply_actions, tick

    state = WorldState(
        food=500.0,
        wood=200.0,
        gold=90.0,
        stone=200.0,
        population=22,
        pop_cap=30,
        age="Feudal Age",
        buildings=["mill", "lumber_camp"],
        villager_queue=[],
        age_up_ticks_remaining=0,
    )
    for _ in range(5):
        actions = decide([], _gamestate_from_world(state), alarm=False)
        state = tick(apply_actions(state, actions))
    assert state.buildings.count("mining_camp") == 1


def test_reactive_tier_alone_reaches_castle_age() -> None:
    """T-544 closed loop: from a fresh Feudal state, with NO LLM in the loop,
    the reactive tier must stand up the gold drop-off and both qualifying
    buildings, bank 800 food + 200 gold, and press the age up. Run 13 reached
    Feudal and then idled for 25 minutes (F-46)."""
    from core import WorldState
    from evaluation.world_sim import apply_actions, tick

    state = WorldState(
        food=200.0,
        wood=200.0,
        gold=90.0,  # run 13's actual Feudal gold
        stone=200.0,
        population=30,
        pop_cap=45,
        age="Feudal Age",
        buildings=["mill", "lumber_camp"],
        villager_queue=[],
        age_up_ticks_remaining=0,
    )
    for _ in range(60):
        actions = decide([], _gamestate_from_world(state), alarm=False)
        state = tick(apply_actions(state, actions))
        if state.age == "Castle Age":
            break
    assert state.age == "Castle Age", f"stuck in {state.age} after {state.turn} turns"
    assert {"mining_camp", "blacksmith", "market"}.issubset(set(state.buildings))
    assert state.buildings.count("market") == 1  # prep stops once each one stands


def test_reactive_age_up_requirements_match_world_sim() -> None:
    """Drift guard (V-4): run 6 proved the sim knew a requirement the agent
    lacked. Pin every age's cost and prereq set across the two."""
    from evaluation.world_sim import AGE_UP_REQUIREMENTS
    from gameplay_agent import reactive

    assert reactive._AGE_UP_REQUIREMENTS.keys() == AGE_UP_REQUIREMENTS.keys()
    for age, requirement in reactive._AGE_UP_REQUIREMENTS.items():
        sim = AGE_UP_REQUIREMENTS[age]
        assert (requirement.next_age, requirement.food, requirement.gold) == (
            sim.next_age,
            sim.food,
            sim.gold,
        )
        assert requirement.prereq_classes == sim.prereq_buildings


def test_reactive_build_constants_match_executor_tables() -> None:
    """Drift guard (V-4 seed): the reactive tier duplicates build keys/costs to
    stay dependency-free — this pins every copy to the executor's tables."""
    from gameplay_agent import executor as ex
    from gameplay_agent import reactive

    for prep in (*reactive._FEUDAL_PREP_BUILDS, *reactive._CASTLE_PREP_BUILDS):
        assert ex.BUILD_KEY_TO_CLASS[prep.build_key] == prep.building_class
        assert ex._BUILD_WOOD_COST[prep.build_key] == prep.wood_cost
    assert ex._BUILD_WOOD_COST["a"] == reactive._FARM_WOOD_COST
    assert ex.BUILD_KEY_TO_CLASS[reactive._FARM_BUILD_KEY] == "farm"
    assert ex.BUILD_KEY_TO_CLASS[reactive._HOUSE_BUILD_KEY] == "house"
    # Every class an age-up waits for must be one some prep rule can build.
    buildable = {
        p.building_class for p in (*reactive._FEUDAL_PREP_BUILDS, *reactive._CASTLE_PREP_BUILDS)
    }
    for requirement in reactive._AGE_UP_REQUIREMENTS.values():
        assert requirement.prereq_classes.issubset(buildable)
    # The reactive house trigger must stay INSIDE the executor's allow band,
    # or every emit would be rejected as ample headroom.
    assert reactive._HOUSE_HEADROOM_TRIGGER <= ex._HOUSE_HEADROOM_MAX
    assert reactive._GAME_POP_CAP_LIMIT == ex._GAME_POP_CAP_LIMIT
    # The executor's order gate backstops the reactive per-age targets (F-38,
    # T-538) — whole-map pin, and the message map must cover the same ages.
    assert reactive._VILLAGER_TARGET_BY_AGE == ex._VILLAGER_ORDER_TARGET_BY_AGE
    assert ex._VILLAGER_ORDER_TARGET_BY_AGE.keys() == ex._NEXT_AGE.keys()


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
