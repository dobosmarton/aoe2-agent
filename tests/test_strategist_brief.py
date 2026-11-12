"""The strategist's facts come from the agent's records and the action catalog."""

from __future__ import annotations

from dataclasses import replace

from gameplay_agent.policy.state import PolicyState
from gameplay_agent.strategist_brief import strategist_brief

_LATE_DARK_AGE = PolicyState(
    food=411,
    wood=836,
    population=32,
    population_cap=35,
    pending_population=1,
    villagers=31,
    idle_count=7,
    villager_jobs={"food": 6, "wood": 18, "gold": 0, "stone": 0},
    buildings_seen=frozenset({"house", "mill", "lumber_camp", "farm"}),
    building_purchases=frozenset({"house"}),
    pending_buildings=frozenset({"house", "barracks"}),
    visible_classes=frozenset({"villager", "tree", "farm"}),
)


def _brief(state: PolicyState = _LATE_DARK_AGE, failures: tuple[str, ...] = ()) -> str:
    return strategist_brief(state, failures)


def test_villagers_are_shown_with_their_jobs_and_idle_count() -> None:
    assert "Villagers: 31 (food 6, wood 18, gold 0, stone 0, idle 7)." in _brief()


def test_the_next_age_cost_is_shown_against_the_bank() -> None:
    assert "Feudal Age costs 500 food (have 411)." in _brief()


def test_a_met_building_requirement_is_called_met() -> None:
    assert "2 standing (lumber_camp, mill), requirement met." in _brief()


def test_a_short_building_requirement_says_how_many_more() -> None:
    one_building = replace(_LATE_DARK_AGE, buildings_seen=frozenset({"house", "mill"}))
    assert "1 standing (mill), 1 more needed." in _brief(one_building)


def test_an_unread_resource_is_not_shown_as_zero() -> None:
    food_unread = replace(_LATE_DARK_AGE, known_resources=frozenset({"wood"}))
    assert "Feudal Age costs 500 food (have not read)." in _brief(food_unread)


def test_an_unread_job_count_is_not_shown_as_zero() -> None:
    wood_only = replace(_LATE_DARK_AGE, villager_jobs={"wood": 18})
    assert "food not read, wood 18" in _brief(wood_only)


def test_a_paid_building_is_not_also_listed_as_unpaid() -> None:
    assert "Placed, payment not yet seen: barracks." in _brief()


def test_gold_and_stone_are_not_gathered_in_the_dark_age() -> None:
    assert "Resources the agent gathers in this age: food, wood." in _brief()


def test_a_farm_is_not_natural_food() -> None:
    assert "Natural food in view (sheep, berries, ...): none." in _brief()


def test_a_paid_building_not_yet_seen_finished_is_listed() -> None:
    assert "Paid for, not yet seen finished: house." in _brief()


def test_actions_show_their_cost_and_requirements() -> None:
    assert "- build_farm: 60 wood; needs mill." in _brief()


def test_actions_of_a_later_age_are_left_out() -> None:
    assert "build_blacksmith" not in _brief()


def test_recent_failures_are_passed_on() -> None:
    brief = _brief(failures=("assign_food: no food target with verified bounds",))
    assert "Recent failed actions: assign_food: no food target with verified bounds." in brief


def test_an_unread_villager_count_is_not_guessed() -> None:
    assert "Villagers: count not read." in _brief(PolicyState(villagers=None))


def test_an_unread_population_is_not_guessed() -> None:
    assert "Population: not read." in _brief(PolicyState(population_known=False))
