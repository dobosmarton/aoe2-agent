"""Qualification checks for the supported game profile, on Windows or macOS.

Run ``python -m gameplay_agent.preflight --profile PROFILE.json`` while the
game is open with a villager or Town Center selected. This cannot prove a key
binding's in-game effect; that remains a separate manual qualification gate.

Nothing here clicks or presses keys. On macOS the cursor moves to the middle of the
game and back is read, to prove that frame pixels and input points line up.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .game_profile import GameProfile, load_profile
from .resource_ocr import (
    calibration_for,
    prepare_hud_pixels,
    read_resource_bar_pixels,
    read_selected_unit_pixels,
)
from .screen import NativeCapture, capture_native
from .window import game_platform, get_game_geometry, get_game_window_rect

if TYPE_CHECKING:
    from .window import GameGeometry

# The band under the menu bar is black; game frames are not.
_BAND_ROWS = 10
_BAND_BRIGHTNESS = 8.0
_CURSOR_SETTLE_SECONDS = 0.1
# Points round to whole input units, so a round trip can land one pixel off.
_ROUND_TRIP_TOLERANCE_PX = 1
_SRGB_NAMES = ("srgb", "internet")


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    # A failed check that isn't blocking is a warning: the run can still go ahead.
    blocking: bool = True


class _PreflightArgs(argparse.Namespace):
    profile: Path
    screenshot: Path | None


def inspect_capture(profile: GameProfile, screenshot: bytes) -> tuple[PreflightCheck, ...]:
    """Validate geometry and HUD signals without trusting model text."""
    import io

    try:
        with Image.open(io.BytesIO(screenshot)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError) as exc:
        return (PreflightCheck("capture", False, f"unreadable image: {type(exc).__name__}"),)
    return _inspect_pixels(profile, rgb)


def inspect_pixels(profile: GameProfile, capture: NativeCapture) -> tuple[PreflightCheck, ...]:
    """Inspect the live frame before any lossy screenshot encoding."""
    return _inspect_pixels(profile, capture.pixels())


def _inspect_pixels(profile: GameProfile, rgb: np.ndarray) -> tuple[PreflightCheck, ...]:
    size = (int(rgb.shape[1]), int(rgb.shape[0]))
    correct_size = size == (profile.capture_width, profile.capture_height)
    checks = [
        PreflightCheck(
            "capture_geometry",
            correct_size,
            f"captured {size[0]}x{size[1]}, profile {profile.capture_width}x{profile.capture_height}",
        )
    ]
    calibration = calibration_for(*size)
    if not correct_size or calibration is None:
        checks.append(PreflightCheck("hud_calibration", False, "no matching qualified crop set"))
        return tuple(checks)
    try:
        readings = read_resource_bar_pixels(
            prepare_hud_pixels(rgb, size), calibration, backend="template"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        checks.append(PreflightCheck("hud_reading", False, f"reader error: {type(exc).__name__}"))
        return tuple(checks)
    core = sum(name in readings for name in ("food", "wood", "gold", "stone"))
    required = core >= 3 and all(
        name in readings for name in ("population", "villagers", "idle_present")
    )
    checks.append(
        PreflightCheck("hud_reading", required, f"decoded fields: {', '.join(sorted(readings))}")
    )
    try:
        selected = read_selected_unit_pixels(rgb, calibration)
    except (OSError, ValueError, RuntimeError):
        selected = None
    checks.append(
        PreflightCheck(
            "selected_object",
            selected in {"villager", "town_center"},
            f"read {selected or 'unknown'}; select a villager or Town Center and rerun",
        )
    )
    return tuple(checks)


def run_preflight(
    profile: GameProfile, screenshot: bytes | None = None
) -> tuple[PreflightCheck, ...]:
    """Check the game window and one capture before an agent run."""
    rect = get_game_window_rect()
    window_ok = rect is not None and rect[2:] == (
        profile.capture_width,
        profile.capture_height,
    )
    checks = [
        PreflightCheck(
            "window",
            window_ok,
            f"window rectangle: {rect or 'not found'}",
        ),
        PreflightCheck(
            "roster",
            profile.roster_verified,
            "recorded player names, colors, and teams must be verified against the lobby",
        ),
        PreflightCheck(
            "hotkeys",
            profile.hotkeys_verified,
            "manual in-game binding/effect verification is required",
        ),
        PreflightCheck(
            "ownership_colors",
            profile.ownership_verified,
            "team-color recognition requires labeled Windows combat screenshots",
        ),
    ]
    macos = game_platform() == "macos"
    if macos:
        checks.extend(_macos_checks(profile))
    frame_width: int | None = None
    if screenshot is None:
        if rect is None:
            return tuple(checks)
        if macos and not _passed(checks, "screen_recording"):
            return tuple(checks)
        capture = capture_native()
        frame_width = capture.width
        checks.extend(inspect_pixels(profile, capture))
        if macos:
            checks.append(_band_check(capture.pixels()))
    else:
        checks.extend(inspect_capture(profile, screenshot))
    if macos:
        checks.append(
            _cursor_round_trip(get_game_geometry(), _passed(checks, "accessibility"), frame_width)
        )
    else:
        checks.append(
            PreflightCheck(
                "capture_to_input_geometry",
                window_ok,
                "capture origin and input window offset agree only when the exact game window "
                "is found",
            )
        )
    return tuple(checks)


def _passed(checks: list[PreflightCheck], name: str) -> bool:
    return any(check.name == name and check.passed for check in checks)


def _macos_checks(profile: GameProfile) -> list[PreflightCheck]:
    from .keyboard_macos import use_layout_keys
    from .window_macos import (
        colour_space,
        input_allowed,
        screen_recording_allowed,
    )

    layout, missing = use_layout_keys()
    colours = colour_space()
    return [
        PreflightCheck(
            "profile_platform",
            profile.platform == "macos",
            f"profile is for {profile.platform}; use a macOS profile on the Mac",
        ),
        PreflightCheck(
            "screen_recording",
            screen_recording_allowed(),
            "allow this terminal in System Settings > Privacy & Security > Screen Recording",
        ),
        PreflightCheck(
            "accessibility",
            input_allowed(),
            "allow this terminal in System Settings > Privacy & Security > Accessibility",
        ),
        PreflightCheck(
            "keyboard_layout",
            not missing,
            f"the {layout.name} layout can't type {' '.join(missing)} without Option; "
            "add and select a layout that can"
            if missing
            else f"keys are sent through the {layout.name} layout",
        ),
        PreflightCheck(
            "colour_profile",
            any(name in colours.lower() for name in _SRGB_NAMES),
            f"display colour profile: {colours}; the detector learned sRGB colours, so "
            "choose Internet & Web (sRGB) in System Settings > Displays",
            blocking=False,
        ),
    ]


def _band_check(rgb: np.ndarray) -> PreflightCheck:
    brightness = float(np.mean(rgb[:_BAND_ROWS]))
    return PreflightCheck(
        "no_black_band",
        brightness >= _BAND_BRIGHTNESS,
        f"top rows' mean brightness {brightness:.1f}; a black band means the frame "
        "still includes the strip under the menu bar",
    )


def _cursor_round_trip(
    geometry: GameGeometry | None, input_allowed: bool, frame_width: int | None
) -> PreflightCheck:
    """Prove that frame pixels and input units line up.

    Two parts. The captured frame's width over the screen's width in input units must
    equal the scale the executor divides by; the full-screen game spans the screen.
    And the cursor, sent to the frame's centre, must read back there, which shows that
    input reaches the system at all.
    """
    name = "capture_to_input_geometry"
    if geometry is None:
        return PreflightCheck(name, False, "the game window wasn't found")
    if not input_allowed:
        return PreflightCheck(name, False, "skipped: this terminal can't move the cursor yet")
    import pyautogui

    screen_width = int(pyautogui.size().width)
    measured = (frame_width or geometry.width) / screen_width
    target = (geometry.width // 2, geometry.height // 2)
    pyautogui.moveTo(*geometry.to_input(*target))
    time.sleep(_CURSOR_SETTLE_SECONDS)
    position = pyautogui.position()
    back = geometry.to_frame(position.x, position.y)
    error = max(abs(back[0] - target[0]), abs(back[1] - target[1]))
    return PreflightCheck(
        name,
        error <= _ROUND_TRIP_TOLERANCE_PX and abs(measured - geometry.scale) < 1e-6,
        f"frame width over screen width is scale {measured:g} and the executor uses "
        f"{geometry.scale:g}; the cursor sent to frame {target} read back at {back}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only supported-profile preflight")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(namespace=_PreflightArgs())
    profile = load_profile(args.profile)
    screenshot = args.screenshot.read_bytes() if args.screenshot is not None else None
    checks = run_preflight(profile, screenshot)
    print(json.dumps([asdict(check) for check in checks], indent=2))
    raise SystemExit(0 if all(check.passed or not check.blocking for check in checks) else 1)


if __name__ == "__main__":
    main()
