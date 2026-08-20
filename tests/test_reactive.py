"""Unit tests for the deterministic policy tier (gameplay_agent/policy/).

Pure functions over fake entity dicts + GameState. No executor / pyautogui.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest
from gameplay_agent.entity_utils import nearest_class_of_kind
from gameplay_agent.memory import GameState
from gameplay_agent.policy.engine import decide as _policy_decide
from gameplay_agent.policy.idle import resolve_idle_target as _resolve_idle_target
from gameplay_agent.policy.state import from_game_state

from tests.factories import make_entity as _ent


def decide(entities: list, state: GameState, alarm: bool) -> list[dict]:
    """Drive the engine from a GameState, as the game loop does."""
    return _policy_decide(entities, from_game_state(state, captured_at=time.monotonic()), alarm)


def _routed_kinds(state: GameState, count: int = 5) -> tuple[str, ...]:
    """The next `count` resources the router would staff, in order.

    The Phase 4.1 successor to the old gather-pattern tuple. Routing interleaves
    rather than blocking, so assert on the ratio and the lead, not the sequence.
    """
    from gameplay_agent.policy import allocation
    from gameplay_agent.policy.engine import registry, wood_bank_target

    policy_state = from_game_state(state, captured_at=time.monotonic())
    target = wood_bank_target(policy_state, registry())
    mix = allocation.for_state(policy_state, None, target)
    jobs: dict[str, int] = dict(policy_state.villager_jobs)
    kinds: list[str] = []
    for _ in range(count):
        kind = allocation.next_kind(mix, jobs)
        jobs = allocation.with_one_more(jobs, kind)
        kinds.append(kind)
    return tuple(kinds)


# Picks needed to resolve a one-slot bias. Too small a sample hides it: at 5,
# both a 3:2 and a 3:3 target land 3 food and 2 wood, and Feudal's 3-way split
# needs more still.
_BIAS_SAMPLE = 10


def _ratio(state: GameState, count: int = 5) -> Counter[str]:
    return Counter(_routed_kinds(state, count))


def _state(
    population: int,
    population_cap: int = 30,
    age: str = "Dark Age",
    idle_present: bool | None = None,
    idle_count: int | None = None,
    idle_streak: int = 0,
    food: int = 200,
    wood: int = 200,
    # Default = Feudal prereqs + mining camp satisfied, so tests about OTHER
    # features aren't polluted by prep build actions (lumber camp in Dark Age,
    # mining camp in Feudal); prep/age-up tests override it.
    buildings: frozenset[str] = frozenset({"mill", "lumber_camp", "mining_camp"}),
    # Default = villager target reached, so no queue order rides along in
    # dispatch/age-up tests; queue tests pass explicit lower orders.
    villagers_ordered: int = 30,
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
        # Prereqs are a Dark Age concern (mining camp present so the castle
        # prep doesn't emit its own build here).
        (12, "Feudal Age", frozenset({"mill", "mining_camp"})),
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


def test_age_up_rule_requires_exactly_the_world_sims_prereqs() -> None:
    """Drift guard (V-4): run 6 proved the sim knew the requirement the agent
    lacked. Asserted behaviorally now that the set lives in the rule trigger."""
    from evaluation.world_sim import FEUDAL_PREREQ_BUILDINGS
    from gameplay_agent.policy.engine import registry

    age_up = next(r for r in registry() if r.id == "age_up_feudal")
    banked = _state(population=22, food=520, buildings=FEUDAL_PREREQ_BUILDINGS)
    assert age_up.matches(from_game_state(banked, captured_at=time.monotonic()))
    for missing in FEUDAL_PREREQ_BUILDINGS:
        short = FEUDAL_PREREQ_BUILDINGS - {missing}
        without_prereq = _state(22, food=520, buildings=short)
        assert not age_up.matches(from_game_state(without_prereq, captured_at=time.monotonic()))


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
    assert _ratio(_state(population=22, food=40, wood=65), 3)["wood"] == 1
    assert _ratio(_state(population=22, food=40, wood=85), 3)["wood"] == 0


# ---------------------------------------------------------------------------
# Goal-driven wood bank target (T-518 → T-527)
# ---------------------------------------------------------------------------


def test_wood_below_farm_target_gets_more_wood() -> None:
    """The bank adds a wood slot; it must not take every slot (F-8/F-21)."""
    banking = _ratio(_state(population=20, wood=50), _BIAS_SAMPLE)
    banked = _ratio(_state(population=20, wood=200), _BIAS_SAMPLE)
    assert banking["wood"] > banked["wood"]
    assert banking["food"] > 0  # never all-wood


def test_lumber_camp_goal_raises_the_wood_target() -> None:
    """Run 8 (F-34): wood plateaued at 65 while the camp cost 100 — the bias
    must keep pulling wood until the CAMP is affordable, not just a farm."""
    needs_camp = _state(population=20, wood=110, buildings=frozenset({"mill"}))
    banked = _state(population=20, wood=125, buildings=frozenset({"mill"}))
    assert (
        _ratio(needs_camp, _BIAS_SAMPLE)["wood"] > _ratio(banked, _BIAS_SAMPLE)["wood"]
    )  # 110 < 100 + margin
    assert _ratio(banked) == Counter({"food": 3, "wood": 2})  # Dark Age 3:2


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
    assert _ratio(state) == Counter({"food": 3, "wood": 2})  # Dark Age 3:2


def test_missing_mill_raises_the_wood_target() -> None:
    """The mill is a Feudal prereq AND the farm unlock, so once the economy is
    established a missing mill banks wood toward it — like the camp (F-41)."""
    needs_mill = _state(population=20, wood=110, buildings=frozenset())
    banked = _state(population=20, wood=125, buildings=frozenset())
    assert (
        _ratio(needs_mill, _BIAS_SAMPLE)["wood"] > _ratio(banked, _BIAS_SAMPLE)["wood"]
    )  # 110 < 100 + margin
    assert _ratio(banked) == Counter({"food": 3, "wood": 2})


def test_missing_mining_camp_raises_the_wood_target() -> None:
    """T-538: in Feudal the camp (gold drop-off) outranks the farm band, same
    arrangement as the Dark Age prereqs."""
    needs_camp = _state(
        population=25, age="Feudal Age", wood=110, buildings=frozenset({"mill", "lumber_camp"})
    )
    banked = _state(
        population=25, age="Feudal Age", wood=300, buildings=frozenset({"mill", "lumber_camp"})
    )
    assert (
        _ratio(needs_camp, _BIAS_SAMPLE)["wood"] > _ratio(banked, _BIAS_SAMPLE)["wood"]
    )  # 110 < 100 + margin


def test_feudal_gold_bias_until_castle_gold_banked() -> None:
    """T-538: below 200 gold in Feudal the rotation leads with gold; once the
    Castle gold is banked the base pattern returns. Dark Age never biases."""
    poor = _state(population=25, age="Feudal Age", wood=300)  # factory gold=100
    assert _ratio(poor, 6)["gold"] > _ratio(poor, 6)["stone"]
    rich = _state(population=25, age="Feudal Age", wood=300)
    rich.resources["gold"] = 250
    assert _ratio(rich) == Counter({"food": 2, "wood": 2, "gold": 1})  # Feudal 2:2:1
    assert _routed_kinds(_state(population=20, wood=300))[0] != "gold"  # Dark Age


def test_wood_bias_outranks_gold_bias() -> None:
    # The mining camp the gold bank needs costs wood — buildings come first.
    state = _state(
        population=25, age="Feudal Age", wood=50, buildings=frozenset({"mill", "lumber_camp"})
    )
    assert _ratio(state, _BIAS_SAMPLE)["wood"] > _ratio(state, _BIAS_SAMPLE)["gold"]


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
    villagers_ordered high) so the closed loop below isolates the Feudal-prep
    BUILD path — the concern F-41 is about.
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


def test_every_build_rule_declares_the_executors_wood_cost() -> None:
    """Drift guard (V-4 successor): a rule's declared `cost` drives reservation,
    so it must equal the executor's table — and this covers rules added later."""
    from gameplay_agent import executor as ex
    from gameplay_agent.policy.engine import registry

    build_rules = [r for r in registry() if r.actions[0].get("type") == "build"]
    assert build_rules, "registry has no build rules — the loader is broken"
    for rule in build_rules:
        key = str(rule.actions[0]["building_key"])
        assert rule.cost.get("wood") == ex._BUILD_WOOD_COST[key], rule.id


def test_the_farm_is_the_only_cost_idle_still_owns() -> None:
    """Every other build cost now comes from the rule that emits it (V-4).

    The farm has no rule — the idle path emits it directly — so this is the one
    copy left, and it must track the executor's table.
    """
    from gameplay_agent import executor as ex
    from gameplay_agent.policy import idle

    assert ex._BUILD_WOOD_COST["a"] == idle._FARM_WOOD_COST
    assert ex.BUILD_KEY_TO_CLASS[idle._FARM_BUILD_KEY] == "farm"
    assert ex._VILLAGER_ORDER_TARGET_BY_AGE.keys() == ex._NEXT_AGE.keys()


def test_house_rule_stays_inside_the_executors_allow_band() -> None:
    """A trigger above _HOUSE_HEADROOM_MAX would have every emit rejected."""
    from gameplay_agent import executor as ex
    from gameplay_agent.policy.engine import registry

    house = next(r for r in registry() if r.id == "house_when_headroom_gone")
    for headroom in range(ex._HOUSE_HEADROOM_MAX + 1):
        state = from_game_state(
            _state(population=30 - headroom, population_cap=30), captured_at=time.monotonic()
        )
        if house.matches(state):
            return
    raise AssertionError("house rule never fires inside the executor's allow band")


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
    """A 3-batch must not dump everyone on one tile — the F-4 blanket bug."""
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("tree", (20, 20))]
    turn = decide(entities, _state(population=22, idle_present=True), alarm=False)
    targets = [a["target_class"] for a in turn if a["type"] == "right_click"]
    assert set(targets) == {"sheep", "tree"}


def test_idle_distribution_follows_the_dark_age_ratio() -> None:
    """3:2 food:wood, whatever order the shortfall router lands them in."""
    assert _ratio(_state(population=22)) == Counter({"food": 3, "wood": 2})


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
