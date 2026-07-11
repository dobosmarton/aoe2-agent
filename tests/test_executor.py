"""Unit tests for gameplay_agent/executor.py.

Tests are organized in four layers:
  1. Pure helpers — coordinate resolution, target lookup, intent re-resolution.
  2. Module state — detected_entities cache + rescan callback registration.
  3. Per-action handlers — pyautogui patched out, asserts on the right calls.
  4. Dispatcher — execute_action's normalization (Action vs dict), unknown
     types, exception handling.

pyautogui is monkeypatched per-test to record calls without invoking real
mouse/keyboard. The window-rect / focus / action-delay side effects are
also stubbed so tests don't sleep or depend on a focused game window.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gameplay_agent.executor import ActionResult

# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


class _FakePyautogui:
    """Records every call so tests can assert what would have happened."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.FAILSAFE = False  # the real module attribute the executor reads
        self.PAUSE = 0.0

    def _record(self, name: str) -> Callable[..., None]:
        def fn(*args: object, **kwargs: object) -> None:
            self.calls.append((name, args, kwargs))

        return fn

    def __getattr__(self, name: str) -> Callable[..., None]:
        # Default: any pyautogui.* call gets recorded as a no-op.
        return self._record(name)

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def fake_pyautogui(monkeypatch: pytest.MonkeyPatch) -> _FakePyautogui:
    """Replace `executor.pyautogui` with a recorder; stub side-effecty deps."""
    fake = _FakePyautogui()
    monkeypatch.setattr(ex, "pyautogui", fake)
    # Default window-rect → no offset; focus check always succeeds.
    monkeypatch.setattr(ex, "get_game_window_rect", lambda: None)
    monkeypatch.setattr(ex, "ensure_game_focused", lambda: True)
    monkeypatch.setattr(ex.config, "action_delay", 0.0, raising=False)
    # Reset module-level state every test so they don't leak.
    ex._detected_entities = []
    ex._window_offset = (0, 0)
    ex._rescan_fn = None
    ex._rescan_full_fn = None
    ex.reset_build_gates()
    return fake


# ---------------------------------------------------------------------------
# Layer 1 — pure helpers
# ---------------------------------------------------------------------------


def test_resolve_target_id_found(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [
        {"id": "sheep_1", "class": "sheep", "center": (100, 200)},
        {"id": "tree_3", "class": "tree", "center": (50, 60)},
    ]
    assert ex._resolve_target_id("sheep_1") == (100, 200)
    assert ex._resolve_target_id("tree_3") == (50, 60)


def test_resolve_target_id_missing_returns_none(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [{"id": "sheep_1", "class": "sheep", "center": (1, 2)}]
    assert ex._resolve_target_id("missing") is None


def test_resolve_target_id_skips_entities_without_center(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex._detected_entities = [{"id": "sheep_1", "class": "sheep"}]  # no center
    assert ex._resolve_target_id("sheep_1") is None


def test_resolve_target_class_returns_first_match(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [
        {"id": "tree_1", "class": "tree", "center": (10, 20)},
        {"id": "tree_2", "class": "tree", "center": (30, 40)},  # ignored
    ]
    assert ex._resolve_target_class("tree") == (10, 20)


def test_resolve_target_class_missing_returns_none(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [{"id": "x", "class": "tree", "center": (1, 2)}]
    assert ex._resolve_target_class("gold") is None


def test_resolve_coords_prefers_target_id(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [{"id": "sheep_1", "class": "sheep", "center": (200, 300)}]
    err, coords = ex._resolve_coords({"target_id": "sheep_1", "x": 0, "y": 0})
    assert err == ""
    assert coords == (200, 300)


def test_resolve_coords_falls_back_to_class(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [{"id": "x", "class": "sheep", "center": (50, 60)}]
    err, coords = ex._resolve_coords({"target_class": "sheep"})
    assert err == ""
    assert coords == (50, 60)


def test_resolve_coords_uses_xy_when_no_targets(fake_pyautogui: _FakePyautogui) -> None:
    err, coords = ex._resolve_coords({"x": 123, "y": 456})
    assert err == ""
    assert coords == (123, 456)


def test_resolve_coords_rejects_placeholder_zero_zero(fake_pyautogui: _FakePyautogui) -> None:
    err, coords = ex._resolve_coords({"x": 0, "y": 0})
    assert "(0, 0) placeholder" in err
    assert coords is None


def test_resolve_coords_target_id_not_found_returns_error(
    fake_pyautogui: _FakePyautogui,
) -> None:
    err, coords = ex._resolve_coords({"target_id": "ghost"})
    assert "target_id 'ghost'" in err
    assert coords is None


def test_resolve_coords_no_input_returns_error(fake_pyautogui: _FakePyautogui) -> None:
    err, coords = ex._resolve_coords({})
    assert "no coordinates" in err
    assert coords is None


def test_translate_adds_window_offset(fake_pyautogui: _FakePyautogui) -> None:
    ex._window_offset = (100, 50)
    assert ex._translate(10, 20) == (110, 70)


def test_re_resolve_from_intent_picks_matching_class(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex._detected_entities = [
        {"id": "tree_1", "class": "tree", "center": (300, 400)},
    ]
    new_x, new_y = ex._re_resolve_from_intent(0, 0, "Send villager to gather from tree")
    assert (new_x, new_y) == (300, 400)


def test_re_resolve_from_intent_skips_actor_classes(fake_pyautogui: _FakePyautogui) -> None:
    """'villager' / 'town_center' appear as the subject, never the target —
    re-resolution must not snap to them."""
    ex._detected_entities = [
        {"id": "villager_1", "class": "villager", "center": (999, 999)},
    ]
    new_x, new_y = ex._re_resolve_from_intent(50, 60, "Send villager to gather wood")
    assert (new_x, new_y) == (50, 60)  # unchanged


def test_re_resolve_from_intent_no_match_returns_original(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex._detected_entities = [{"id": "tree_1", "class": "tree", "center": (1, 2)}]
    assert ex._re_resolve_from_intent(7, 8, "build a house") == (7, 8)


# ---------------------------------------------------------------------------
# Layer 2 — module state setters / getters
# ---------------------------------------------------------------------------


def test_set_and_get_detected_entities_with_dict(fake_pyautogui: _FakePyautogui) -> None:
    raw = [{"id": "s1", "class": "sheep"}]
    ex.set_detected_entities(raw)
    assert ex.get_detected_entities() == raw


def test_set_detected_entities_calls_to_dict(fake_pyautogui: _FakePyautogui) -> None:
    """Entities with a `to_dict()` method are converted; raw dicts pass through."""

    class FakeEntity:
        def to_dict(self) -> dict[str, object]:
            return {"id": "from_obj", "class": "tree"}

    ex.set_detected_entities([FakeEntity(), {"id": "raw", "class": "sheep"}])
    got = ex.get_detected_entities()
    assert got[0] == {"id": "from_obj", "class": "tree"}
    assert got[1] == {"id": "raw", "class": "sheep"}


def test_clear_detected_entities(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_detected_entities([{"id": "x"}])
    ex.clear_detected_entities()
    assert ex.get_detected_entities() == []


def test_set_rescan_fn_registers_callback(fake_pyautogui: _FakePyautogui) -> None:
    async def cb() -> None: ...

    ex.set_rescan_fn(cb)
    assert ex._rescan_fn is cb


# ---------------------------------------------------------------------------
# Layer 3 — per-action handlers (pyautogui patched out)
# ---------------------------------------------------------------------------


def test_handle_click_translates_and_clicks(fake_pyautogui: _FakePyautogui) -> None:
    ex._window_offset = (100, 50)
    result = _run(ex._handle_click({"x": 10, "y": 20}, "ordinary click"))
    assert result.success is True
    assert ("click", (110, 70), {}) in fake_pyautogui.calls


def test_handle_click_returns_failure_when_coords_unresolvable(
    fake_pyautogui: _FakePyautogui,
) -> None:
    result = _run(ex._handle_click({"target_id": "missing"}, "click thing"))
    assert result.success is False
    assert "missing" in result.detail
    # And no actual click was attempted.
    assert "click" not in fake_pyautogui.names()


def test_handle_click_build_intent_triggers_retry_clicks(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """Build/place intents click N+1 times (initial + retries) and finish
    with a right-click cancel on the original."""
    _run(ex._handle_click({"x": 200, "y": 300}, "Place a house"))
    click_count = sum(1 for c in fake_pyautogui.names() if c == "click")
    right_clicks = sum(1 for c in fake_pyautogui.names() if c == "rightClick")
    assert click_count == 1 + ex.BUILD_RETRY_ATTEMPTS
    assert right_clicks == 1


def test_handle_right_click_translates_and_calls(fake_pyautogui: _FakePyautogui) -> None:
    # Coords must sit on the map (not the HUD margins) or the play-area guard drops them.
    ex._detected_entities = [{"id": "s1", "class": "sheep", "center": (500, 600)}]
    result = _run(ex._handle_right_click({"target_id": "s1"}, "gather wood"))
    assert result.success is True
    assert ("rightClick", (500, 600), {}) in fake_pyautogui.calls


def test_handle_right_click_re_resolves_when_no_target(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """Without target_id/target_class, the executor should re-resolve from
    intent — overriding stale x/y coords from the LLM."""
    ex._detected_entities = [{"id": "tree_1", "class": "tree", "center": (700, 800)}]
    _run(ex._handle_right_click({"x": 1, "y": 2}, "send villager to tree"))
    assert ("rightClick", (700, 800), {}) in fake_pyautogui.calls


def test_handle_press_simple_key(fake_pyautogui: _FakePyautogui) -> None:
    result = _run(ex._handle_press({"key": "h"}, "go to TC"))
    assert result.success is True
    assert ("press", ("h",), {}) in fake_pyautogui.calls


def test_handle_press_with_modifiers_uses_hotkey(fake_pyautogui: _FakePyautogui) -> None:
    _run(ex._handle_press({"key": "1", "modifiers": ["ctrl"]}, "select group"))
    assert ("hotkey", ("ctrl", "1"), {}) in fake_pyautogui.calls
    # Plain `press` should NOT have been called.
    assert "press" not in fake_pyautogui.names()


def test_handle_press_with_rescan_invokes_rescan_fn(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    rescan_called = False

    async def fake_rescan() -> None:
        nonlocal rescan_called
        rescan_called = True

    ex._rescan_fn = fake_rescan
    # Eliminate the post-key settle delay so the test is fast.
    monkeypatch.setattr(ex, "RESCAN_SETTLE_DELAY", 0.0)
    _run(ex._handle_press({"key": ".", "rescan": True}, "select idle"))
    assert rescan_called is True


# ---------------------------------------------------------------------------
# House headroom gate — build_rejection + the _handle_build reject path
# ---------------------------------------------------------------------------


def test_build_rejection_allows_without_snapshot(fake_pyautogui: _FakePyautogui) -> None:
    # No population reading yet → never block on missing data.
    assert ex.build_rejection("q") is None


def test_build_rejection_blocks_house_with_ample_headroom(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex.set_hud_snapshot(10, 30, {})  # 20 headroom — another house is wasted wood
    reason = ex.build_rejection("q")
    assert reason is not None and "headroom" in reason


def test_build_rejection_allows_house_near_cap(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(26, 30, {})  # headroom 4 = the gate boundary, allowed
    assert ex.build_rejection("q") is None


def test_build_rejection_blocks_house_at_game_cap(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(199, 200, {})
    reason = ex.build_rejection("q")
    assert reason is not None and "maximum" in reason


def test_build_rejection_headroom_gate_is_house_only(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 30, {})  # 20 headroom blocks houses, nothing else
    assert ex.build_rejection("w") is None  # mill
    ex.record_confirmed_buildings(["mill"])
    assert ex.build_rejection("a") is None  # farm (prereq satisfied)


def test_handle_build_rejects_house_with_headroom(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 30, {})
    result = _run(ex._handle_build({"building_key": "q"}, "Build house to increase pop cap"))
    assert result.success is False and "headroom" in result.detail
    assert fake_pyautogui.calls == []  # rejected before any key was pressed


# ---------------------------------------------------------------------------
# Build gates — prerequisite (farm needs a seen mill) + wood cost
# ---------------------------------------------------------------------------


def test_farm_rejected_without_mill(fake_pyautogui: _FakePyautogui) -> None:
    # No mill ever detected: the farm menu entry doesn't exist — silent no-op.
    reason = ex.build_rejection("a")
    assert reason is not None and "mill" in reason


def test_farm_allowed_once_mill_seen(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_detected_entities([{"id": "mill_0", "class": "mill", "center": (100, 100)}])
    ex.set_detected_entities([])  # camera moved away — evidence must persist
    assert ex.build_rejection("a") is None


def test_farm_rejected_when_wood_short_even_with_mill(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    ex.set_hud_snapshot(10, 15, {"wood": 30})
    reason = ex.build_rejection("a")
    assert reason is not None and "60 wood" in reason and "30" in reason


def test_cost_gate_blocks_unaffordable_mill(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 15, {"wood": 50})
    reason = ex.build_rejection("w")
    assert reason is not None and "100 wood" in reason


def test_cost_gate_allows_when_resources_unknown(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    assert ex.build_rejection("a") is None  # no snapshot → never block on missing data


def test_record_confirmed_buildings_ignores_non_gate_classes(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex.record_confirmed_buildings(["sheep", "villager", "town_center"])
    assert ex._buildings_confirmed == set()


def test_reset_build_gates_clears_evidence(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    ex.set_hud_snapshot(10, 15, {"wood": 500})
    ex.reset_build_gates()
    assert ex.build_rejection("a") is not None  # mill evidence gone


# ---------------------------------------------------------------------------
# Effect-level placement verification (_handle_click with building_key)
# ---------------------------------------------------------------------------


def _zero_build_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "BUILD_SETTLE_DELAY", 0.0)
    monkeypatch.setattr(ex, "BUILD_RETRY_DELAY", 0.0)
    monkeypatch.setattr(ex, "RESCAN_SETTLE_DELAY", 0.0)


def test_place_click_unconfirmed_stays_success_and_goes_pending(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foundations aren't detectable — an unseen building is NOT a failure (the
    false 'failed' caused run 2's duplicate mill); it queues for wood-delta
    settlement instead."""
    _zero_build_delays(monkeypatch)
    ex.set_hud_snapshot(10, 15, {"wood": 200})  # baseline for the pending entry

    async def rescan_sees_nothing() -> None:
        ex._detected_entities = []

    ex._rescan_fn = rescan_sees_nothing
    result = _run(
        ex._handle_click({"x": 500, "y": 600, "building_key": "q"}, "Place building (house)")
    )
    assert result.success is True and "not visually confirmed" in result.detail
    assert [p.building_class for p in ex._pending_placements] == ["house"]


def test_place_click_preexisting_building_does_not_vouch(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old farm inside the radius must not verify a new one — the count has
    to INCREASE."""
    _zero_build_delays(monkeypatch)
    ex.set_hud_snapshot(10, 15, {"wood": 200})
    old_farm = {"id": "farm_0", "class": "farm", "center": (505, 610)}
    ex._detected_entities = [old_farm]

    async def rescan_same_farm() -> None:
        ex._detected_entities = [old_farm]

    ex._rescan_fn = rescan_same_farm
    _run(ex._handle_click({"x": 500, "y": 600, "building_key": "a"}, "Place building (farm)"))
    assert [p.building_class for p in ex._pending_placements] == ["farm"]  # unconfirmed


def test_pending_placement_confirmed_by_wood_spend(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 15, {"wood": 200})
    ex._note_pending_placement("w")  # mill, 100 wood
    # Next snapshot: wood dropped by ~the cost → the purchase happened.
    ex.set_hud_snapshot(10, 15, {"wood": 95})
    assert ex._pending_placements == []
    assert ex.build_rejection("a") is None  # confirmed mill unlocks farms


def test_pending_placement_missing_when_wood_kept(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 15, {"wood": 200})
    ex._note_pending_placement("w")
    # Wood went UP (gathering, nothing spent) → the placement never happened.
    ex.set_hud_snapshot(10, 15, {"wood": 230})
    assert ex._pending_placements == []
    assert "mill" not in ex._buildings_confirmed


def test_pending_placement_waits_out_stale_readings(fake_pyautogui: _FakePyautogui) -> None:
    ex.set_hud_snapshot(10, 15, {"wood": 200})
    ex._note_pending_placement("w")
    # An identical reading is stale OCR, not evidence — the entry survives.
    ex.set_hud_snapshot(10, 15, {"wood": 200})
    assert len(ex._pending_placements) == 1
    ex.set_hud_snapshot(10, 15, {"wood": 90})  # fresh reading settles it
    assert ex._pending_placements == [] and "mill" in ex._buildings_confirmed


def test_place_click_succeeds_and_records_when_building_lands(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    _zero_build_delays(monkeypatch)

    async def rescan_sees_mill() -> None:
        ex._detected_entities = [{"id": "mill_0", "class": "mill", "center": (505, 610)}]

    ex._rescan_fn = rescan_sees_mill
    result = _run(
        ex._handle_click({"x": 500, "y": 600, "building_key": "w"}, "Place building (mill)")
    )
    assert result.success is True
    # A verified placement is prerequisite evidence: farms are now buildable.
    assert ex.build_rejection("a") is None


def test_place_click_unverifiable_without_rescan_gets_benefit_of_doubt(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    _zero_build_delays(monkeypatch)
    result = _run(
        ex._handle_click({"x": 500, "y": 600, "building_key": "q"}, "Place building (house)")
    )
    assert result.success is True  # no rescan callback → cannot verify → allow


def test_handle_build_rejects_farm_without_mill(fake_pyautogui: _FakePyautogui) -> None:
    result = _run(ex._handle_build({"building_key": "a"}, "Build farm for food"))
    assert result.success is False and "mill" in result.detail
    assert fake_pyautogui.calls == []


def test_handle_drag_emits_moveto_then_drag(fake_pyautogui: _FakePyautogui) -> None:
    _run(
        ex._handle_drag(
            {"start_x": 10, "start_y": 20, "end_x": 50, "end_y": 60},
            "select group",
        )
    )
    names = fake_pyautogui.names()
    assert "moveTo" in names and "drag" in names
    # drag deltas should be end - start
    drag_call = next(c for c in fake_pyautogui.calls if c[0] == "drag")
    assert drag_call[1] == (40, 40)


def test_handle_scroll_with_coords(fake_pyautogui: _FakePyautogui) -> None:
    _run(ex._handle_scroll({"clicks": 3, "x": 10, "y": 20}, "zoom in"))
    scroll_call = next(c for c in fake_pyautogui.calls if c[0] == "scroll")
    assert scroll_call[1] == (3,)
    assert scroll_call[2] == {"x": 10, "y": 20}


def test_handle_scroll_without_coords(fake_pyautogui: _FakePyautogui) -> None:
    _run(ex._handle_scroll({"clicks": -2}, "zoom out"))
    scroll_call = next(c for c in fake_pyautogui.calls if c[0] == "scroll")
    assert scroll_call[1] == (-2,)
    assert scroll_call[2] == {}


def test_handle_wait_sleeps_proportional_to_ms(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _run(ex._handle_wait({"ms": 500}, "settle"))
    assert sleeps == [0.5]


def test_handle_detect_calls_rescan_full_when_set(
    fake_pyautogui: _FakePyautogui,
) -> None:
    called = False

    async def full() -> None:
        nonlocal called
        called = True

    ex._rescan_full_fn = full
    result = _run(ex._handle_detect({}, "force scan"))
    assert result.success is True
    assert called is True


def test_handle_detect_returns_failure_when_unavailable(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex._rescan_full_fn = None
    result = _run(ex._handle_detect({}, "force scan"))
    assert result.success is False


# ---------------------------------------------------------------------------
# Layer 4 — execute_action dispatcher
# ---------------------------------------------------------------------------


def test_execute_action_rejects_unknown_type(fake_pyautogui: _FakePyautogui) -> None:
    """Unknown action type is rejected early by validate_action (returns
    'invalid action format'). The dispatcher's own 'unknown action type'
    branch is only reachable via a Pydantic model whose type lacks a handler
    — see test_execute_action_dispatcher_handles_missing_handler below."""
    result = _run(ex.execute_action({"type": "fly_to_moon", "intent": "win the game"}))
    assert result.success is False
    assert "invalid action format" in result.detail


def test_execute_action_dispatcher_handles_missing_handler(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a valid Action type somehow lacks a handler, the dispatcher
    returns a structured failure rather than KeyErroring."""
    from gameplay_agent.models import PressAction

    # Remove 'press' from the dispatch table to simulate the inconsistency.
    monkeypatch.delitem(ex._ACTION_HANDLERS, "press")
    action = PressAction(type="press", key="h", intent="x")
    result = _run(ex.execute_action(action))
    assert result.success is False
    assert "unknown action type 'press'" in result.detail


def test_execute_action_normalizes_dict_input(fake_pyautogui: _FakePyautogui) -> None:
    """A raw dict is validated through the Action union and dispatched."""
    result = _run(ex.execute_action({"type": "press", "key": "h", "intent": "TC"}))
    assert result.success is True
    assert ("press", ("h",), {}) in fake_pyautogui.calls


def test_execute_action_rejects_invalid_dict(fake_pyautogui: _FakePyautogui) -> None:
    """Action dict that fails Pydantic validation returns a failure result."""
    result = _run(ex.execute_action({"type": "press"}))  # missing required `key`
    assert result.success is False


def test_execute_action_accepts_pydantic_model(fake_pyautogui: _FakePyautogui) -> None:
    from gameplay_agent.models import PressAction

    action = PressAction(type="press", key="h", intent="go to TC")
    result = _run(ex.execute_action(action))
    assert result.success is True
    assert ("press", ("h",), {}) in fake_pyautogui.calls


def test_execute_action_catches_handler_exceptions(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handler raising an unexpected error becomes a structured failure
    result rather than crashing the loop."""

    async def boom(_a: dict, _i: str) -> ActionResult:
        raise RuntimeError("kaboom")

    monkeypatch.setitem(ex._ACTION_HANDLERS, "press", boom)
    result = _run(ex.execute_action({"type": "press", "key": "h", "intent": "x"}))
    assert result.success is False
    assert "kaboom" in result.detail


def test_execute_action_refreshes_window_offset_from_rect(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each call refreshes _window_offset from get_game_window_rect()."""
    monkeypatch.setattr(ex, "get_game_window_rect", lambda: (200, 100, 1920, 1080))
    _run(ex.execute_action({"type": "click", "x": 10, "y": 20, "intent": "click"}))
    assert ex._window_offset == (200, 100)
    # And the click was performed in screen-absolute coords.
    assert ("click", (210, 120), {}) in fake_pyautogui.calls


def test_execute_actions_runs_each_in_order(fake_pyautogui: _FakePyautogui) -> None:
    actions = [
        {"type": "press", "key": "h", "intent": "TC"},
        {"type": "press", "key": "q", "intent": "queue"},
    ]
    results = _run(ex.execute_actions(actions))
    assert len(results) == 2
    assert all(r.success for r in results)
    press_keys = [c[1][0] for c in fake_pyautogui.calls if c[0] == "press"]
    assert press_keys == ["h", "q"]


def test_build_steps_sequence() -> None:
    """build_steps yields select-villager -> econ-menu -> building-key -> placement click."""
    steps = ex.build_steps("w", "build mill", (700, 400))
    assert [s["type"] for s in steps] == ["press", "press", "press", "click"]
    assert steps[2]["key"] == "w"  # building_key selects the structure
    assert (steps[3]["x"], steps[3]["y"]) == (700, 400)  # placement passed through
