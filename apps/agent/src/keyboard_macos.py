"""Type the character the game expects, whatever the Mac keyboard layout.

The game reads the character a key types in the active layout, but pyautogui sends
the key that types it on a US keyboard. On a Hungarian layout its "z" is the key
labelled Y, so the game saw "y". At startup the agent asks macOS which key types
each character in the active layout, and points pyautogui's table at those keys.

The layout is read through Carbon's Text Input Sources, which must run on the main
thread: call `use_layout_keys` at startup, not from a worker.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, MutableMapping

log = structlog.stdlib.get_logger()

# Characters the agent's hotkeys use: building and unit letters, control groups,
# and the idle-villager keys.
AGENT_KEYS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.,")

_CARBON = "/System/Library/Frameworks/Carbon.framework/Carbon"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_KEY_ACTION_DISPLAY = 3  # kUCKeyActionDisplay
_SHIFT_STATE = 2  # shiftKey (1 << 9) shifted right by 8, as UCKeyTranslate wants.
_NO_DEAD_KEYS = 1  # 1 << kUCKeyTranslateNoDeadKeysBit
_MAIN_KEY_CODES = range(0x33)  # The typing keys; the keypad starts at 0x41.

# A key code, and whether Shift must be held to type the character.
KeyStroke = tuple[int, bool]


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    keys: dict[str, KeyStroke]  # The lowest key code that types each character.


def read_layout() -> Layout:
    """The active keyboard layout, from macOS. Main thread only."""
    import objc  # macOS only; the rest of this module is plain Python.

    carbon = ctypes.CDLL(_CARBON)
    core = ctypes.CDLL(_CORE_FOUNDATION)
    carbon.TISCopyCurrentKeyboardLayoutInputSource.restype = ctypes.c_void_p
    carbon.TISGetInputSourceProperty.restype = ctypes.c_void_p
    carbon.TISGetInputSourceProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    carbon.LMGetKbdType.restype = ctypes.c_uint8
    carbon.UCKeyTranslate.restype = ctypes.c_int32
    carbon.UCKeyTranslate.argtypes = [
        ctypes.c_void_p,  # Layout data.
        ctypes.c_uint16,  # Key code.
        ctypes.c_uint16,  # Key action.
        ctypes.c_uint32,  # Modifier state.
        ctypes.c_uint32,  # Keyboard type.
        ctypes.c_uint32,  # Options.
        ctypes.POINTER(ctypes.c_uint32),  # Dead-key state.
        ctypes.c_ulong,  # Buffer length.
        ctypes.POINTER(ctypes.c_ulong),  # Characters written.
        ctypes.c_void_p,  # Buffer.
    ]
    core.CFDataGetBytePtr.restype = ctypes.c_void_p
    core.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    core.CFRelease.argtypes = [ctypes.c_void_p]

    # ctypes calls return whatever `restype` says; the casts tell the checker.
    source = cast("int | None", carbon.TISCopyCurrentKeyboardLayoutInputSource())
    if not source:
        raise RuntimeError("macOS reported no keyboard layout")
    try:
        name_key = ctypes.c_void_p.in_dll(carbon, "kTISPropertyLocalizedName")
        data_key = ctypes.c_void_p.in_dll(carbon, "kTISPropertyUnicodeKeyLayoutData")
        name_ref = cast("int", carbon.TISGetInputSourceProperty(source, name_key))
        name = str(objc.objc_object(c_void_p=name_ref))  # pyright: ignore[reportAttributeAccessIssue]
        data = cast("int | None", carbon.TISGetInputSourceProperty(source, data_key))
        if not data:
            raise RuntimeError(f"the {name} layout has no key table")
        table = cast("int", core.CFDataGetBytePtr(data))
        keyboard_type = cast("int", carbon.LMGetKbdType())

        def typed(code: int, state: int) -> str:
            dead = ctypes.c_uint32(0)
            length = ctypes.c_ulong(0)
            buffer = (ctypes.c_uint16 * 4)()
            status = cast(
                "int",
                carbon.UCKeyTranslate(
                    table,
                    code,
                    _KEY_ACTION_DISPLAY,
                    state,
                    keyboard_type,
                    _NO_DEAD_KEYS,
                    ctypes.byref(dead),
                    len(buffer),
                    ctypes.byref(length),
                    buffer,
                ),
            )
            units = cast("list[int]", buffer[: length.value])
            return "".join(map(chr, units)) if status == 0 else ""

        return Layout(name, layout_keys(typed))
    finally:
        core.CFRelease(source)


def layout_keys(typed: Callable[[int, int], str]) -> dict[str, KeyStroke]:
    """Which key types each character, given what each key types with and without Shift.

    Unshifted keys win over shifted ones, and lower key codes over higher.
    """
    keys: dict[str, KeyStroke] = {}
    for shift in (False, True):
        for code in _MAIN_KEY_CODES:
            character = typed(code, _SHIFT_STATE if shift else 0)
            if len(character) == 1 and character.isprintable() and not character.isspace():
                keys.setdefault(character, (code, shift))
    return keys


def remap(
    mapping: MutableMapping[str, int | None],
    keys: dict[str, KeyStroke],
    needs_shift: Callable[[str], bool],
) -> list[str]:
    """Point each single-character entry of pyautogui's table at the layout's key.

    pyautogui decides Shift by US rules (`needs_shift`) and then sends the table's
    key. A character whose Shift need differs in this layout, or that it can't type,
    is removed, so pyautogui drops it instead of typing something else. Returns the
    characters removed.
    """
    removed: list[str] = []
    for character in [name for name in mapping if len(name) == 1 and not name.isspace()]:
        stroke = keys.get(character)
        if stroke is None or stroke[1] != needs_shift(character):
            mapping[character] = None
            removed.append(character)
        else:
            mapping[character] = stroke[0]
    return removed


def untypeable(keys: Iterable[str], removed: Iterable[str]) -> list[str]:
    """The agent's keys that the layout can't type the way pyautogui sends them."""
    gone = set(removed)
    return sorted(key for key in keys if key in gone)


def use_layout_keys() -> tuple[Layout, list[str]]:
    """Make pyautogui type through the active layout. Returns the layout and the
    agent keys it can't type. Main thread only."""
    import importlib

    import pyautogui

    table = cast(
        "MutableMapping[str, int | None]",
        importlib.import_module("pyautogui._pyautogui_osx").keyboardMapping,
    )
    layout = read_layout()
    removed = remap(table, layout.keys, pyautogui.isShiftCharacter)
    missing = untypeable(AGENT_KEYS, removed)
    log.info(
        "keyboard_layout_applied",
        layout=layout.name,
        untypeable_agent_keys=missing,
        characters_dropped=len(removed),
    )
    return layout, missing


__all__ = [
    "AGENT_KEYS",
    "Layout",
    "layout_keys",
    "read_layout",
    "remap",
    "untypeable",
    "use_layout_keys",
]
