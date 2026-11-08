"""Mac input: point-scaled clicks, a held button, no system shortcuts, and no input
while another app is in front. The Windows arithmetic must not change."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pytest
from gameplay_agent import executor as ex
from gameplay_agent import game_loop, window
from gameplay_agent.game_profile import load_profile
from gameplay_agent.turn_phases import get_ground_commands

from tests.window_fakes import VM_GAME, FakeWindowBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

T = TypeVar("T")
_MAC_PROFILE = (
    Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.macos.example.json"
)
# The middle of the 3024x1898 game area, in points: (1512, 949 + 66) / 2, rounded.
_MAC_CENTRE = (756, 508)


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


class _Input:
    """Records every pyautogui call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str) -> Callable[..., None]:
        def record(*args: object, **kwargs: object) -> None:
            self.calls.append((name, args, kwargs))

        return record

    def named(self, *names: str) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
        return [call for call in self.calls if call[0] in names]


@pytest.fixture
def mouse(monkeypatch: pytest.MonkeyPatch) -> _Input:
    recorder = _Input()
    monkeypatch.setattr(ex, "pyautogui", recorder)
    monkeypatch.setattr(ex.config, "action_delay", 0.0, raising=False)
    monkeypatch.setattr(ex, "MAC_CLICK_HOLD_S", 0.0)
    monkeypatch.setattr(ex, "MAC_SCROLL_GAP_S", 0.0)
    monkeypatch.setattr(ex, "_window_offset", (0, 0))
    monkeypatch.setattr(ex, "_input_scale", 1.0)
    monkeypatch.setattr(ex, "_platform", "none")
    monkeypatch.setattr(ex, "_modifier_map", {})
    ex.clear_detected_entities()
    ex.reset_build_gates()
    return recorder


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> FakeWindowBackend:
    backend = FakeWindowBackend()
    monkeypatch.setattr(window, "_backend", backend)
    return backend


# ---------------------------------------------------------------------------
# Windows stays as it was
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [(0, 0), (200, 100), (-8, 31)])
def test_scale_one_translation_is_the_old_offset_sum(
    mouse: _Input, monkeypatch: pytest.MonkeyPatch, offset: tuple[int, int]
) -> None:
    monkeypatch.setattr(ex, "_window_offset", offset)
    assert ex._translate(1000, 800) == (1000 + offset[0], 800 + offset[1])


def test_the_vm_window_parks_the_cursor_where_it_always_did(
    mouse: _Input, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(window, "_backend", FakeWindowBackend(platform="windows", game=VM_GAME))
    assert _run(ex.execute_action({"type": "click", "x": 1000, "y": 800, "intent": "x"})).success
    assert mouse.named("click") == [("click", (1200, 900), {})]
    # The old hard-coded move: rect[0] + 1512, rect[1] + 836.
    assert mouse.named("moveTo")[-1] == ("moveTo", (1712, 936), {})


def test_windows_sends_ctrl_chords_unchanged(
    mouse: _Input, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(window, "_backend", FakeWindowBackend(platform="windows", game=VM_GAME))
    ex.set_modifier_map([("ctrl", "command")])  # Ignored off macOS.
    action = {"type": "press", "key": "h", "modifiers": ["ctrl"], "intent": "x"}
    assert _run(ex.execute_action(action)).success
    assert mouse.named("hotkey") == [("hotkey", ("ctrl", "h"), {})]


def test_windows_ground_commands_still_zoom_by_scrolling() -> None:
    assert [c["type"] for c in get_ground_commands(1)] == ["scroll", "press", "press"]
    assert [c["type"] for c in get_ground_commands(1, relative_zoom=False)] == ["press", "press"]


# ---------------------------------------------------------------------------
# macOS: points, held clicks, fixed zoom
# ---------------------------------------------------------------------------


def test_a_mac_click_is_held_in_points_then_the_cursor_parks_mid_game(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    assert _run(ex.execute_action({"type": "click", "x": 1000, "y": 800, "intent": "x"})).success
    assert mouse.named("click") == []
    assert mouse.named("mouseDown", "mouseUp") == [
        ("mouseDown", (500, 433), {"button": "left"}),
        ("mouseUp", (500, 433), {"button": "left"}),
    ]
    assert mouse.named("moveTo")[-1] == ("moveTo", _MAC_CENTRE, {})


def test_a_mac_right_click_is_held_with_the_right_button(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    assert _run(ex.execute_action({"type": "right_click", "x": 1000, "y": 800, "intent": "x"}))
    assert mouse.named("mouseDown") == [("mouseDown", (500, 433), {"button": "right"})]


def test_a_mac_drag_moves_in_points(mouse: _Input, mac: FakeWindowBackend) -> None:
    action = {"type": "drag", "start_x": 1000, "start_y": 800, "end_x": 1200, "end_y": 900}
    assert _run(ex.execute_action({**action, "intent": "x"})).success
    assert mouse.named("moveTo")[0] == ("moveTo", (500, 433), {})
    assert mouse.named("drag") == [("drag", (100, 50), {"duration": 0.2})]


def test_a_mac_scroll_is_refused_so_the_zoom_stays_fixed(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    result = _run(ex.execute_action({"type": "scroll", "clicks": 3, "intent": "x"}))
    assert not result.success
    assert result.detail == ex.MAC_ZOOM_FIXED
    assert mouse.named("scroll") == []


def test_the_zoom_recipe_goes_all_the_way_out_then_in_one_step_at_a_time(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    assert _run(ex.set_absolute_zoom(4)).success
    assert mouse.calls[0] == ("moveTo", _MAC_CENTRE, {})
    steps = [call[1][0] for call in mouse.named("scroll")]
    assert steps == [-1] * ex.MAC_FULL_ZOOM_OUT_STEPS + [1] * 4


def test_the_zoom_recipe_stops_when_the_game_leaves_the_front(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    mac.front_answers = [True] * 6 + [False]  # The first answer is the start check.
    result = _run(ex.set_absolute_zoom(4))
    assert not result.success
    assert len(mouse.named("scroll")) == 5


# ---------------------------------------------------------------------------
# macOS: modifiers and system shortcuts
# ---------------------------------------------------------------------------


def test_a_mapped_modifier_reaches_the_game(mouse: _Input, mac: FakeWindowBackend) -> None:
    ex.set_modifier_map([("ctrl", "command")])
    action = {"type": "press", "key": "b", "modifiers": ["ctrl"], "intent": "barracks"}
    assert _run(ex.execute_action(action)).success
    assert mouse.named("hotkey") == [("hotkey", ("command", "b"), {})]


@pytest.mark.parametrize(
    ("modifiers", "key", "mapping"),
    [
        (["ctrl"], "h", [("ctrl", "command")]),  # Mapped into Cmd+H, which hides the game.
        (["command"], "q", []),
        (["cmd", "shift"], "q", []),  # Contains Cmd+Q, and logs out besides.
        (["ctrl"], "up", []),  # Mission Control.
        (["control"], "space", []),  # Switches the input source.
        ([], "f11", []),  # Show Desktop.
    ],
)
def test_system_shortcuts_are_never_sent(
    mouse: _Input,
    mac: FakeWindowBackend,
    modifiers: list[str],
    key: str,
    mapping: list[tuple[str, str]],
) -> None:
    ex.set_modifier_map(mapping)
    result = _run(ex.execute_action({"type": "press", "key": key, "modifiers": modifiers}))
    assert not result.success
    assert "system shortcut" in result.detail
    assert mouse.named("hotkey", "press") == []


# ---------------------------------------------------------------------------
# macOS: the game must be in front
# ---------------------------------------------------------------------------


def test_no_input_while_another_app_is_in_front(mouse: _Input, mac: FakeWindowBackend) -> None:
    mac.front = False
    results = _run(ex.execute_actions([{"type": "press", "key": "h", "intent": "x"}]))
    assert [(r.success, r.detail) for r in results] == [(False, ex.GAME_NOT_IN_FRONT)]
    assert mouse.calls == []
    assert mac.focus_calls == 0  # Switching away stops the agent; it never steals focus back.


def test_input_stops_mid_batch_when_the_game_leaves_the_front(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    mac.front_answers = [True, True, False]  # Batch check, first action, second action.
    actions = [{"type": "press", "key": "h"}, {"type": "press", "key": "q"}]
    results = _run(ex.execute_actions(actions))
    assert [r.success for r in results] == [True, False]
    assert [call[1] for call in mouse.named("press")] == [("h",)]


def test_no_window_means_no_input(mouse: _Input, mac: FakeWindowBackend) -> None:
    mac.game = None
    mac.front = False
    result = _run(ex.execute_action({"type": "click", "x": 1000, "y": 800, "intent": "x"}))
    assert result.detail == ex.GAME_NOT_IN_FRONT
    assert mouse.calls == []


# ---------------------------------------------------------------------------
# macOS startup
# ---------------------------------------------------------------------------


def test_mac_startup_brings_the_game_forward_then_applies_the_profile(
    mouse: _Input, mac: FakeWindowBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    zooms: list[int] = []

    async def zoom(steps: int) -> ex.ActionResult:
        zooms.append(steps)
        return ex.ActionResult(True, "ok")

    monkeypatch.setattr(game_loop, "set_absolute_zoom", zoom)
    monkeypatch.setattr(game_loop, "_popup_open", lambda: False)
    profile = load_profile(_MAC_PROFILE).model_copy(
        update={"zoom_steps": 6, "modifier_map": {"ctrl": "command"}}
    )
    assert _run(game_loop._prepare_macos_game(profile)) is None
    assert mac.focus_calls == 1
    assert zooms == [6]
    assert ex._modifier_map == {"ctrl": "command"}


def test_mac_startup_stops_when_the_game_cannot_be_brought_forward(
    mouse: _Input, mac: FakeWindowBackend
) -> None:
    mac.front = False
    assert _run(game_loop._prepare_macos_game(None)) == "lost_focus"
    assert mouse.calls == []


def test_mac_startup_stops_while_a_menu_is_open(
    mouse: _Input, mac: FakeWindowBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(game_loop, "_popup_open", lambda: True)
    assert _run(game_loop._prepare_macos_game(None)) == "menu_open"
    assert mouse.calls == []  # No zoom: the scroll would land on the menu.


def test_a_mac_key_press_first_releases_stuck_modifiers(
    mouse: _Input, mac: FakeWindowBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent import keyboard_macos

    order: list[str] = []
    monkeypatch.setattr(keyboard_macos, "release_stuck_modifiers", lambda: order.append("release"))
    monkeypatch.setattr(mouse, "press", lambda key: order.append(f"press {key}"), raising=False)
    assert _run(ex.execute_action({"type": "press", "key": "h", "intent": "x"})).success
    assert order == ["release", "press h"]
