"""Unit tests for S5 deterministic reactive tier (gameplay_agent/reactive.py).

Pure functions over fake entity dicts + GameState. No executor / pyautogui.
"""

from __future__ import annotations

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
) -> GameState:
    return GameState(
        population=population,
        population_cap=population_cap,
        current_age=age,
        idle_present=idle_present,
        idle_count=idle_count,
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


def test_no_idle_dispatch_without_resources() -> None:
    actions = decide([_ent("villager")], _state(population=22, idle_present=True), alarm=False)
    assert actions == []


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
