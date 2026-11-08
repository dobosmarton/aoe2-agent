"""Keys are sent through the active layout, so the game reads the intended character."""

from __future__ import annotations

import sys

import pytest
from gameplay_agent.keyboard_macos import AGENT_KEYS, Layout, layout_keys, remap, untypeable

SHIFT = 2
# A Hungarian Mac keyboard, as UCKeyTranslate reports it: Y and Z swap places, 0 sits
# left of 1, and "/" needs Shift.
HUNGARIAN = {
    (6, 0): "y",
    (16, 0): "z",
    (10, 0): "0",
    (29, 0): "ö",
    (4, 0): "h",
    (47, 0): ".",
    (6, SHIFT): "Y",
    (16, SHIFT): "Z",
    (21, SHIFT): "!",
    (26, SHIFT): "/",
    (36, 0): "\r",  # Return: not a character to remap.
    (50, 0): "í",
    (50, SHIFT): "Í",
}
# pyautogui's US table: the key codes that type these characters on a US keyboard.
US = {"y": 16, "z": 6, "0": 29, "h": 4, ".": 47, "!": 18, "/": 44, "@": 19, "shift": 56}


def _typed(code: int, state: int) -> str:
    return HUNGARIAN.get((code, state), "")


def _us_shift(character: str) -> bool:
    return character.isupper() or character in '~!@#$%^&*()_+{}|:"<>?'


def test_the_layout_table_prefers_unshifted_keys() -> None:
    keys = layout_keys(_typed)
    assert keys["y"] == (6, False)
    assert keys["z"] == (16, False)
    assert keys["Y"] == (6, True)
    assert keys["/"] == (26, True)
    assert "\r" not in keys


def test_y_and_z_are_sent_to_the_keys_that_type_them() -> None:
    mapping: dict[str, int | None] = dict(US)
    remap(mapping, layout_keys(_typed), _us_shift)
    assert (mapping["y"], mapping["z"], mapping["0"]) == (6, 16, 10)
    assert mapping["h"] == 4
    assert mapping["!"] == 21  # pyautogui holds Shift for "!", and so does the layout.
    assert mapping["shift"] == 56  # Named keys sit in the same place on every layout.


def test_characters_the_layout_types_differently_are_dropped_not_mistyped() -> None:
    mapping: dict[str, int | None] = dict(US)
    removed = remap(mapping, layout_keys(_typed), _us_shift)
    assert mapping["/"] is None  # US types it unshifted; Hungarian needs Shift.
    assert mapping["@"] is None  # Hungarian needs Option.
    assert sorted(removed) == ["/", "@"]


def test_the_agent_keys_it_cannot_type_are_named() -> None:
    assert untypeable(AGENT_KEYS, ["/", "@", "q"]) == ["q"]
    assert untypeable(AGENT_KEYS, ["/"]) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the Mac's own layout")
def test_this_macs_layout_types_the_agent_keys() -> None:
    from gameplay_agent.keyboard_macos import read_layout

    layout = read_layout()
    assert isinstance(layout, Layout)
    assert layout.name
    assert len(layout.keys) > 20
