"""Unit tests for S5 deterministic reactive tier (gameplay_agent/reactive.py).

Pure functions over fake entity dicts + GameState. No executor / pyautogui.
"""

from __future__ import annotations

from gameplay_agent.memory import GameState
from gameplay_agent.reactive import _nearest_resource_class, decide


def _ent(cls: str, center: tuple[float, float] = (0.0, 0.0)) -> dict:
    return {"class": cls, "id": f"{cls}_0", "center": center, "confidence": 0.9}


def _state(population: int, population_cap: int = 30, age: str = "Dark Age") -> GameState:
    return GameState(population=population, population_cap=population_cap, current_age=age)


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
# Idle-villager assignment
# ---------------------------------------------------------------------------


def test_idle_assign_targets_nearest_food() -> None:
    entities = [
        _ent("town_center", (0, 0)),
        _ent("sheep", (10, 10)),
        _ent("berry_bush", (500, 500)),
    ]
    # population at cap so the queue rule stays quiet — isolate the idle rule.
    actions = decide(entities, _state(population=22), alarm=False)
    assert _types(actions) == ["press", "right_click"]
    assert actions[1]["target_class"] == "sheep"


def test_no_idle_assign_without_resources() -> None:
    actions = decide([_ent("villager")], _state(population=22), alarm=False)
    assert actions == []


# ---------------------------------------------------------------------------
# Alarm + nearest-resource + determinism
# ---------------------------------------------------------------------------


def test_returns_empty_on_alarm() -> None:
    entities = [_ent("sheep", (10, 10))]
    assert decide(entities, _state(population=5), alarm=True) == []


def test_nearest_resource_prefers_food_over_wood() -> None:
    entities = [_ent("tree", (1, 1)), _ent("sheep", (900, 900))]
    assert _nearest_resource_class(entities) == "sheep"


def test_nearest_resource_ranks_by_distance_to_tc() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10)), _ent("boar", (900, 900))]
    assert _nearest_resource_class(entities) == "sheep"


def test_nearest_resource_none_when_absent() -> None:
    assert _nearest_resource_class([_ent("villager")]) is None


def test_decide_is_deterministic() -> None:
    entities = [_ent("town_center", (0, 0)), _ent("sheep", (10, 10))]
    state = _state(population=10)
    assert decide(entities, state, alarm=False) == decide(entities, state, alarm=False)
