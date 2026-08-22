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
    ex.observe_hud(10, 30, {})  # 20 headroom — another house is wasted wood
    reason = ex.build_rejection("q")
    assert reason is not None and "headroom" in reason


def test_build_rejection_allows_house_near_cap(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(26, 30, {})  # headroom 4 = the gate boundary, allowed
    assert ex.build_rejection("q") is None


def test_build_rejection_blocks_house_at_game_cap(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(199, 200, {})
    reason = ex.build_rejection("q")
    assert reason is not None and "maximum" in reason


def test_build_rejection_headroom_gate_is_house_only(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 30, {})  # 20 headroom blocks houses, nothing else
    assert ex.build_rejection("w") is None  # mill
    ex.record_confirmed_buildings(["mill"])
    assert ex.build_rejection("a") is None  # farm (prereq satisfied)


def test_handle_build_rejects_house_with_headroom(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 30, {})
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


def test_detection_sightings_never_unlock_farms(fake_pyautogui: _FakePyautogui) -> None:
    """Run 9 (F-36): a PERSISTENT phantom mill beat the old 3-frame threshold
    and 14 outposts got built through the unlocked farm slot. No sighting
    count is proof — only a purchase (ledger / verified placement) gates."""
    mill_frame = [{"id": "mill_0", "class": "mill", "center": (100, 100)}]
    for _ in range(10):  # persists far past any plausible threshold
        ex.set_detected_entities(mill_frame)
    assert ex.build_rejection("a") is not None  # farm still gated
    assert ex.build_rejection("w") is None  # and the REAL mill is still buildable
    assert ex.sighted_buildings() == frozenset({"mill"})  # reported, not trusted


def test_purchase_confirmed_mill_unlocks_farms(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    assert ex.build_rejection("a") is None


def test_farm_rejected_when_wood_short_even_with_mill(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    ex.observe_hud(10, 15, {"wood": 30})
    reason = ex.build_rejection("a")
    assert reason is not None and "60 wood" in reason and "30" in reason


def test_cost_gate_blocks_unaffordable_mill(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 50})
    reason = ex.build_rejection("w")
    assert reason is not None and "100 wood" in reason


def test_cost_gate_allows_when_resources_unknown(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    assert ex.build_rejection("a") is None  # no snapshot → never block on missing data


def test_record_confirmed_buildings_ignores_non_gate_classes(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex.record_confirmed_buildings(["sheep", "villager", "town_center"])
    assert ex._build_gates.buildings_confirmed == set()


def test_unique_buildings_not_rebuilt(fake_pyautogui: _FakePyautogui) -> None:
    """One mill / lumber camp is enough — the Feudal prep re-emits its build
    every turn and relies on this gate to stop once one stands."""
    ex.record_confirmed_buildings(["mill"])
    reason = ex.build_rejection("w")
    assert reason is not None and "already built" in reason
    # Farms are NOT unique — many are the whole point.
    ex.observe_hud(10, 15, {"wood": 500})
    assert ex.build_rejection("a") is None


def test_unique_building_pending_blocks_double_build(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 300})
    ex._note_pending_placement("r")  # lumber camp placed, settlement pending
    reason = ex.build_rejection("r")
    assert reason is not None and "pending" in reason


def test_confirmed_buildings_accessor(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill", "sheep"])  # non-buildings filtered
    assert ex.confirmed_buildings() == frozenset({"mill"})


def test_pending_placement_counts_accessor(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("a")
    ex._note_pending_placement("a")
    assert ex.pending_placement_counts() == {"farm": 2}
    ex.observe_hud(10, 15, {"wood": 70})  # both spends visible → both settle
    assert ex.pending_placement_counts() == {}


def test_reset_build_gates_clears_evidence(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    ex.observe_hud(10, 15, {"wood": 500})
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
    ex.observe_hud(10, 15, {"wood": 200})  # baseline for the pending entry

    async def rescan_sees_nothing() -> None:
        ex._detected_entities = []

    ex._rescan_fn = rescan_sees_nothing
    result = _run(
        ex._handle_click({"x": 500, "y": 600, "building_key": "q"}, "Place building (house)")
    )
    assert result.success is True and "not visually confirmed" in result.detail
    assert [p.building_class for p in ex._build_gates.pending_placements] == ["house"]


def test_place_click_preexisting_building_does_not_vouch(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old farm inside the radius must not verify a new one — the count has
    to INCREASE."""
    _zero_build_delays(monkeypatch)
    ex.observe_hud(10, 15, {"wood": 200})
    old_farm = {"id": "farm_0", "class": "farm", "center": (505, 610)}
    ex._detected_entities = [old_farm]

    async def rescan_same_farm() -> None:
        ex._detected_entities = [old_farm]

    ex._rescan_fn = rescan_same_farm
    _run(ex._handle_click({"x": 500, "y": 600, "building_key": "a"}, "Place building (farm)"))
    assert [p.building_class for p in ex._build_gates.pending_placements] == ["farm"]  # unconfirmed


def test_pending_placement_confirmed_by_wood_spend(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("w")  # mill, 100 wood
    # Next snapshot: wood dropped by ~the cost → the purchase happened.
    ex.observe_hud(10, 15, {"wood": 95})
    assert ex._build_gates.pending_placements == []
    assert ex.build_rejection("a") is None  # confirmed mill unlocks farms


def test_pending_placement_missing_when_wood_kept(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("w")
    # Wood went UP (gathering, nothing spent) → the placement never happened.
    ex.observe_hud(10, 15, {"wood": 230})
    assert ex._build_gates.pending_placements == []
    assert "mill" not in ex._build_gates.buildings_confirmed


def test_pending_placement_waits_out_stale_readings(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("w")
    # An identical reading is stale OCR, not evidence — the entry survives.
    ex.observe_hud(10, 15, {"wood": 200})
    assert len(ex._build_gates.pending_placements) == 1
    ex.observe_hud(10, 15, {"wood": 90})  # fresh reading settles it
    assert (
        ex._build_gates.pending_placements == [] and "mill" in ex._build_gates.buildings_confirmed
    )


def test_one_wood_drop_confirms_at_most_one_pending(fake_pyautogui: _FakePyautogui) -> None:
    """Run 3 (F-17): one 160→8 drop settled BOTH pending entries. Confirmed
    spend is deducted per shared baseline, so the second pending is judged
    against what the drop has left."""
    ex.observe_hud(10, 15, {"wood": 160})
    ex._note_pending_placement("w")  # mill, 100 wood
    ex._note_pending_placement("r")  # lumber camp, 100 wood — same baseline
    ex.observe_hud(10, 15, {"wood": 8})  # a single purchase's drop
    assert ex._build_gates.buildings_confirmed == {"mill"}  # FIFO winner only
    assert ex._build_gates.pending_placements == []


def test_budget_covers_two_genuine_purchases(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("w")  # mill, 100 wood
    ex._note_pending_placement("a")  # farm, 60 wood — same baseline
    ex.observe_hud(10, 15, {"wood": 30})  # both spends landed (+income slack)
    assert {"mill", "farm"} <= ex._build_gates.buildings_confirmed
    assert ex._build_gates.pending_placements == []


def test_pendings_with_different_baselines_settle_independently(
    fake_pyautogui: _FakePyautogui,
) -> None:
    ex.observe_hud(10, 15, {"wood": 200})
    ex._note_pending_placement("w")
    ex.observe_hud(10, 15, {"wood": 95})  # settles the mill
    ex._note_pending_placement("r")  # fresh baseline 95 — no deduction carryover
    ex.observe_hud(10, 15, {"wood": 0})
    assert {"mill", "lumber_camp"} <= ex._build_gates.buildings_confirmed


def _observe_wood(wood: int) -> None:
    ex.observe_hud(10, 15, {"wood": wood})


def _run_missing_settlements(building_key: str, count: int, wood: int) -> int:
    """Note `count` placements that each vanish (wood only rises); returns wood."""
    for _ in range(count):
        ex._note_pending_placement(building_key)
        wood += 5  # income, no spend → the placement never happened
        _observe_wood(wood)
    return wood


def test_missing_streak_suppresses_the_build(fake_pyautogui: _FakePyautogui) -> None:
    """T-530 (run 9, F-37): a vanishing farm was retried 32 times, each attempt
    buying an unintended outpost — a streak of missing settlements now blocks
    the class with a teaching reason instead."""
    ex.record_confirmed_buildings(["mill"])
    _observe_wood(200)
    _run_missing_settlements("a", count=3, wood=200)
    reason = ex.build_rejection("a")
    assert reason is not None and "suppressed" in reason


def test_suppression_expires_after_window(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    _observe_wood(200)
    wood = _run_missing_settlements("a", count=3, wood=200)
    for _ in range(5):  # _MISSING_SUPPRESS_SNAPSHOTS quiet turns pass
        wood += 5
        _observe_wood(wood)
    assert ex.build_rejection("a") is None  # one retry allowed again


def test_confirmed_purchase_clears_missing_streak(fake_pyautogui: _FakePyautogui) -> None:
    ex.record_confirmed_buildings(["mill"])
    _observe_wood(200)
    wood = _run_missing_settlements("a", count=2, wood=200)
    ex._note_pending_placement("a")
    wood -= 60
    _observe_wood(wood)  # a real purchase settles → streak resets
    _run_missing_settlements("a", count=2, wood=wood)
    assert ex.build_rejection("a") is None  # 2 misses after a success ≠ streak of 3


def test_income_masked_purchase_still_confirms(fake_pyautogui: _FakePyautogui) -> None:
    """Run 13 (F-45/T-537): a 30-villager economy gathered +140 wood across one
    settlement, so the raw delta judged every real purchase MISSING and the
    breaker locked out five building classes. Estimated income is deducted from
    the observed delta before judging. A farm, not a house — houses settle on
    the population cap now."""
    _observe_wood(0)
    _observe_wood(140)  # clean window → income estimate 140/snapshot
    ex._note_pending_placement("a")  # farm, 60 wood, baseline 140
    ex.observe_hud(28, 30, {"wood": 220})  # +80 observed = +140 income, -60 spend
    assert "farm" in ex._build_gates.buildings_confirmed


def test_income_alone_does_not_confirm_a_vanished_placement(
    fake_pyautogui: _FakePyautogui,
) -> None:
    _observe_wood(0)
    _observe_wood(140)
    ex._note_pending_placement("a")
    ex.observe_hud(28, 30, {"wood": 285})  # income only — nothing was spent
    assert "farm" not in ex._build_gates.buildings_confirmed


def test_income_estimate_frozen_while_placement_pending(fake_pyautogui: _FakePyautogui) -> None:
    """A window containing a spend would drag the estimate down and re-open
    the false-missing hole — only clean windows update the EMA."""
    _observe_wood(0)
    _observe_wood(10)  # clean window → estimate 10
    ex._note_pending_placement("w")
    _observe_wood(300)  # polluted window: the pending settles here
    assert ex._build_gates.wood_income_per_snapshot == 10


def test_income_credit_scales_with_stale_snapshots(fake_pyautogui: _FakePyautogui) -> None:
    """Stale-OCR retries accumulate several windows of income before the
    reading moves; the eventual delta must be credited for all of them."""
    _observe_wood(0)
    _observe_wood(50)  # estimate 50/snapshot
    ex._note_pending_placement("w")  # mill, 100 wood, baseline 50
    _observe_wood(50)  # stale reading — entry survives
    _observe_wood(50)  # stale again
    _observe_wood(100)  # 3 windows x 50 income, -100 spend
    assert "mill" in ex._build_gates.buildings_confirmed


def test_verified_placement_lifts_suppression(fake_pyautogui: _FakePyautogui) -> None:
    """T-537 amnesty (run 13, F-45): visual verification proves the build
    path works, so it must clear the T-530 streak exactly like a wood-delta
    confirmation — a class stayed suppressed after it was seen standing."""
    ex.record_confirmed_buildings(["mill"])
    _observe_wood(200)
    _run_missing_settlements("a", count=3, wood=200)
    assert ex.build_rejection("a") is not None  # suppressed
    ex.record_confirmed_buildings(["farm"])  # the verified-placement path
    assert ex.build_rejection("a") is None


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


# ---------------------------------------------------------------------------
# Villager-order ledger (T-531, F-38)
# ---------------------------------------------------------------------------


def test_queue_villager_orders_and_presses(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 15, {"wood": 200, "food": 200})
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert result.success
    assert ex.villagers_ordered() == 5  # the 4 starting villagers + this order
    press_keys = [c[1][0] for c in fake_pyautogui.calls if c[0] == "press"]
    assert press_keys == ["h", "q"]


def test_queue_villager_rejected_at_order_target(fake_pyautogui: _FakePyautogui) -> None:
    """Run 11 (F-38): the brake must fire on ORDERS — the delivered population
    lags by the TC queue depth and over-delivered 40 villagers."""
    ex._build_gates.villagers_ordered = 30
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert not result.success and "target" in result.detail
    assert all(c[0] != "press" for c in fake_pyautogui.calls)  # no keystrokes spent


def test_villager_target_and_message_follow_the_age(fake_pyautogui: _FakePyautogui) -> None:
    """T-538 (run 13): the flat Dark Age 30 overruled the reactive Feudal 35
    and the rejection kept teaching "bank for the Feudal Age" while IN it."""
    ex.observe_age("Feudal Age")
    ex.observe_hud(30, 45, {"wood": 200, "food": 200})
    ex._build_gates.villagers_ordered = 30
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert result.success  # 30 < the Feudal target of 35
    ex._build_gates.villagers_ordered = 35
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert not result.success and "Castle Age" in result.detail


def test_villager_target_uncapped_past_the_age_map(fake_pyautogui: _FakePyautogui) -> None:
    # Castle+ has no order target — only the food gate applies.
    ex.observe_age("Castle Age")
    ex.observe_hud(40, 60, {"wood": 200, "food": 200})
    ex._build_gates.villagers_ordered = 50
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert result.success


def test_queue_villager_rejected_without_food(fake_pyautogui: _FakePyautogui) -> None:
    """A q press with < 50 food no-ops in-game — reject instead, so the ledger
    never counts an order the TC never received."""
    ex.observe_hud(10, 15, {"wood": 200, "food": 40})
    result = _run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert not result.success and "food" in result.detail
    assert ex.villagers_ordered() == 4


def test_starting_villagers_match_initial_population(fake_pyautogui: _FakePyautogui) -> None:
    from gameplay_agent.memory import INITIAL_POPULATION

    assert ex._STARTING_VILLAGERS == INITIAL_POPULATION  # drift guard (V-4)
    assert ex.villagers_ordered() == INITIAL_POPULATION  # fresh-game ledger


# ---------------------------------------------------------------------------
# Stale-coordinate guard (T-525, F-33)
# ---------------------------------------------------------------------------


def test_raw_coords_click_after_camera_move_refused(fake_pyautogui: _FakePyautogui) -> None:
    """A literal-x/y click after a camera-moving press points at pre-jump
    terrain (run 8: villagers walked to random places) — refused, not clicked."""
    results = _run(
        ex.execute_actions(
            [
                {"type": "press", "key": ".", "intent": "select idle"},
                {"type": "right_click", "x": 500, "y": 500, "intent": "send"},
            ]
        )
    )
    assert results[0].success and not results[1].success
    assert "target_class" in results[1].detail  # teaches the fix
    assert all(c[0] != "rightClick" for c in fake_pyautogui.calls)


def test_targeted_click_after_camera_move_executes(fake_pyautogui: _FakePyautogui) -> None:
    ex._detected_entities = [{"id": "sheep_0", "class": "sheep", "center": (400, 300)}]
    results = _run(
        ex.execute_actions(
            [
                {"type": "press", "key": ".", "intent": "select idle"},
                {"type": "right_click", "target_class": "sheep", "intent": "send"},
            ]
        )
    )
    assert all(r.success for r in results)  # resolved from the (fresh) cache


def test_raw_coords_click_without_camera_move_executes(fake_pyautogui: _FakePyautogui) -> None:
    results = _run(ex.execute_actions([{"type": "click", "x": 500, "y": 500, "intent": "ui"}]))
    assert results[0].success


def test_resolve_coords_auto_placement_resolves_at_click_time(
    fake_pyautogui: _FakePyautogui, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ex, "default_build_placement", lambda _key="": (321, 654))
    detail, coords = ex._resolve_coords({"auto_placement": True})
    assert (detail, coords) == ("", (321, 654))


def test_build_steps_sequence() -> None:
    """build_steps: select villager (rescan) -> econ menu -> building key ->
    auto-placed click -> select TC (leave the UI clean so later keys can't land
    in a leaked menu — runs 6-7 built phantom outposts that way; escape here
    opened the game menu when nothing needed canceling, run 8 F-32)."""
    steps = ex.build_steps("w", "build mill")
    assert [s["type"] for s in steps] == ["press", "press", "press", "click", "press"]
    assert steps[0]["rescan"] is True  # '.' moves the camera → refresh entities
    assert steps[2]["key"] == "w"  # building_key selects the structure
    assert steps[3]["auto_placement"] is True  # placement resolved at click time (F-33)
    assert "x" not in steps[3]  # no pre-computed spot survives the camera jump
    assert steps[4]["key"] == "h"  # UI-state hygiene without the game menu


def test_build_selects_by_click_when_nothing_is_idle(fake_pyautogui: _FakePyautogui) -> None:
    """'.' is a no-op with nothing idle, so the leftover TC eats the next 'q'."""
    ex.observe_hud(10, 30, {}, idle_present=False)
    assert ex.build_steps("w", "build mill")[0]["target_class"] == "villager"


def test_build_presses_dot_when_a_villager_is_idle(fake_pyautogui: _FakePyautogui) -> None:
    """An idle villager is the better builder, and '.' re-centers the camera (F-33)."""
    ex.observe_hud(10, 30, {}, idle_present=True)
    assert ex.build_steps("w", "build mill")[0]["key"] == "."


def test_selection_mode_is_recorded_when_the_step_is_built(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """Not re-derived at settlement: the pipeline runs a build one turn after it
    is planned, and `_sync_turn_state` refreshes idle_present in between."""
    ex.observe_hud(10, 30, {"wood": 200}, idle_present=False)
    ex.build_steps("w", "build mill")  # planned in this turn: selects by click
    ex.observe_hud(10, 30, {"wood": 200}, idle_present=True)  # next turn's sync
    ex._note_pending_placement("w", point=(11, 22))  # the build finally runs
    assert ex._build_gates.pending_placements[0].selected_by == "click"


# ---------------------------------------------------------------------------
# House settlement — the population cap, not the wood delta
# ---------------------------------------------------------------------------
#
# A house costs 25 wood against a 20-wood slack, so the whole wood margin is 5
# wood of an ESTIMATED income. Run 2026_08_21_2 built 6 houses (cap 5→15→20→25
# →30→35) while the wood test reported 9 confirmed and 21 missing. The cap moves
# by exactly 5 per completed house and OCR reads it every turn.


def _place_house(cap: int, population: int = 10) -> None:
    """Note one house against a HUD showing `population`/`cap`."""
    ex.observe_hud(population, cap, {"wood": 200})
    ex._note_pending_placement("q")


def test_a_cap_rise_confirms_a_house_even_when_wood_rose(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """The exact case the wood test gets wrong: income hides the 25-wood spend."""
    _place_house(cap=20)
    ex.observe_hud(10, 25, {"wood": 400})  # cap +5, wood UP
    assert "house" in ex._build_gates.buildings_confirmed


def test_an_unmoved_cap_leaves_the_house_missing(fake_pyautogui: _FakePyautogui) -> None:
    """Asserts the streak, not the absence of a confirmation: only a JUDGED
    miss sets it, so an entry stuck pending forever cannot pass this."""
    _place_house(cap=20)
    for _ in range(ex._HOUSE_SETTLE_ATTEMPTS + 1):
        ex.observe_hud(10, 20, {"wood": 400})
    assert ex._build_gates.missing_streaks["house"] == 1


def test_a_house_waits_out_its_construction_time(fake_pyautogui: _FakePyautogui) -> None:
    """A house raises no cap until it finishes, so an unmoved cap is not a miss."""
    _place_house(cap=20)
    ex.observe_hud(10, 20, {"wood": 400})
    assert ex._build_gates.pending_placements  # still waiting, not judged


def test_a_cap_jump_leaves_the_unpaid_house_pending(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """A +10 jump is 2 houses, not 3 — the cap analogue of F-17's two mills."""
    ex.observe_hud(10, 20, {"wood": 200})
    for _ in range(3):
        ex._note_pending_placement("q")
    ex.observe_hud(10, 30, {"wood": 200})  # cap +10
    assert len(ex._build_gates.pending_placements) == 1


def test_a_non_house_still_settles_on_wood(fake_pyautogui: _FakePyautogui) -> None:
    """The cap path is house-only; every other class keeps the wood delta."""
    _observe_wood(200)
    ex._note_pending_placement("w")  # mill, 100 wood
    _observe_wood(100)
    assert "mill" in ex._build_gates.buildings_confirmed


def test_houses_are_not_suppressed_while_pop_capped(fake_pyautogui: _FakePyautogui) -> None:
    """A house is the only way out of a pop cap, so the pause is a deadlock —
    run 2026_08_21_2 sat at 35/35 for the last 10 minutes with houses blocked."""
    for _ in range(ex._MISSING_STREAK_LIMIT + 1):
        _place_house(cap=20, population=20)  # housed: 20/20
        for _ in range(ex._HOUSE_SETTLE_ATTEMPTS + 1):
            ex.observe_hud(20, 20, {"wood": 400})
    # The streak proves the guard returned early — the misses were judged.
    assert ex._build_gates.missing_streaks.get("house", 0) == 0
    assert "house" not in ex._build_gates.suppressed_until


# ---------------------------------------------------------------------------
# Research — the feedback a raw press never had
# ---------------------------------------------------------------------------
#
# Run 2026_08_21_2 pressed the age-up key 10 times over 4 minutes with 3x the
# resources banked. Every press "succeeded" because a keystroke always lands;
# nothing could tell a working button from a greyed-out one.


def _research(tech: str, before: dict[str, int]) -> None:
    """Note one pending research against a HUD showing `before`."""
    ex.observe_hud(10, 30, before)
    ex._note_pending_research(tech, ex._TECHS[tech])


def test_a_matching_resource_drop_confirms_the_research(fake_pyautogui: _FakePyautogui) -> None:
    _research("castle_age", {"food": 900, "gold": 300})
    ex.observe_hud(10, 30, {"food": 100, "gold": 100})
    assert "castle_age" in ex._build_gates.researched


def test_no_resource_drop_reports_the_research_missing(fake_pyautogui: _FakePyautogui) -> None:
    """The greyed-out button: resources kept RISING while the key was pressed."""
    _research("castle_age", {"food": 900, "gold": 300})
    for _ in range(ex._RESEARCH_SETTLE_ATTEMPTS + 1):
        ex.observe_hud(10, 30, {"food": 2600, "gold": 900})
    assert "castle_age" in ex._build_gates.research_blocked_until


def test_an_unmoved_reading_waits_rather_than_failing(fake_pyautogui: _FakePyautogui) -> None:
    """Identical readings are stale OCR, not a refusal."""
    _research("loom", {"food": 500, "gold": 300})
    ex.observe_hud(10, 30, {"food": 500, "gold": 300})
    assert ex._build_gates.pending_research


def test_a_failed_research_is_refused_next_time(fake_pyautogui: _FakePyautogui) -> None:
    """The whole point: one failure, then a reason — not 10 blind retries."""
    _research("castle_age", {"food": 900, "gold": 300})
    for _ in range(ex._RESEARCH_SETTLE_ATTEMPTS + 1):
        ex.observe_hud(10, 30, {"food": 2600, "gold": 900})
    reason = ex.research_rejection("castle_age")
    assert reason is not None and "greyed out" in reason


def test_a_blocked_research_becomes_retryable(fake_pyautogui: _FakePyautogui) -> None:
    """The block must expire. castle_age fails for want of 2 Feudal buildings;
    once the agent builds them, a permanent refusal is the same deadlock one
    level up."""
    _research("castle_age", {"food": 900, "gold": 300})
    for _ in range(ex._RESEARCH_SETTLE_ATTEMPTS + 1):
        ex.observe_hud(10, 30, {"food": 2600, "gold": 900})
    for _ in range(ex._MISSING_SUPPRESS_SNAPSHOTS):
        ex.observe_hud(10, 30, {"food": 2600, "gold": 900})
    assert ex.research_rejection("castle_age") is None


def test_a_half_updated_reading_stays_undecided(fake_pyautogui: _FakePyautogui) -> None:
    """Food fell by the full 800, gold's reading has not caught up. The age-up
    was paid for — reporting it missing here blocked it for the whole game."""
    _research("castle_age", {"food": 900, "gold": 300})
    ex.observe_hud(10, 30, {"food": 100, "gold": 300})
    assert ex._build_gates.pending_research


def test_a_pending_research_is_not_re_pressed(fake_pyautogui: _FakePyautogui) -> None:
    _research("castle_age", {"food": 900, "gold": 300})
    reason = ex.research_rejection("castle_age")
    assert reason is not None and "awaiting HUD settlement" in reason


def test_an_unaffordable_research_is_refused(fake_pyautogui: _FakePyautogui) -> None:
    ex.observe_hud(10, 30, {"food": 100, "gold": 10})
    reason = ex.research_rejection("castle_age")
    assert reason is not None and "costs 800 food" in reason


def test_research_steps_go_to_the_building_then_press_its_key(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """Ctrl-Z goes to the Lumber Camp; `q` is the upgrade slot in its panel."""
    steps = ex.research_steps("double_bit_axe", "faster wood")
    assert [(s["key"], s.get("modifiers", [])) for s in steps] == [("z", ["ctrl"]), ("q", [])]


# ---------------------------------------------------------------------------
# Build menus — the same placement machinery for military and advanced
# ---------------------------------------------------------------------------


def test_the_military_menu_opens_before_its_building_key(
    fake_pyautogui: _FakePyautogui,
) -> None:
    """Barracks is w→q. Without the menu the same `q` would build a House."""
    steps = ex.build_steps("q", "barracks", menu=ex.MILITARY_MENU)
    assert [s.get("key") for s in steps[:3]] == [".", "w", "q"]


def test_the_econ_menu_stays_the_default(fake_pyautogui: _FakePyautogui) -> None:
    steps = ex.build_steps("w", "mill")
    assert [s.get("key") for s in steps[:3]] == [".", "q", "w"]


def test_the_same_key_means_different_buildings_per_menu() -> None:
    assert ex.building_class("q", "w") == "mill"
    assert ex.building_class("w", "w") == "archery_range"


@pytest.mark.parametrize("cls", ["market", "barracks", "archery_range", "stable"])
def test_every_placeable_building_can_become_gate_evidence(cls: str) -> None:
    """A class the agent can place but cannot PROVE can never count toward the
    Castle Age's two-building requirement. The gate set was econ-only before."""
    assert cls in ex._GATE_BUILDING_CLASSES


def test_a_building_the_agent_cannot_place_is_still_filtered() -> None:
    """The set is evidence the agent generated itself, not anything detected."""
    assert "town_center" not in ex._GATE_BUILDING_CLASSES
