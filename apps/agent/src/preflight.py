"""Read-only Windows qualification checks for the supported game profile.

Run ``python -m gameplay_agent.preflight --profile PROFILE.json`` while the
game is open with a villager or Town Center selected. This cannot prove a key
binding's in-game effect; that remains a separate manual qualification gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .game_profile import GameProfile, load_profile
from .resource_ocr import calibration_for, read_resource_bar, read_selected_unit
from .screen import capture_screenshot
from .window import get_game_window_rect


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


class _PreflightArgs(argparse.Namespace):
    profile: Path
    screenshot: Path | None


def inspect_capture(profile: GameProfile, screenshot: bytes) -> tuple[PreflightCheck, ...]:
    """Validate geometry and HUD signals without trusting model text."""
    import io

    try:
        with Image.open(io.BytesIO(screenshot)) as image:
            size = image.size
    except (OSError, ValueError) as exc:
        return (PreflightCheck("capture", False, f"unreadable image: {type(exc).__name__}"),)
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
        readings = read_resource_bar(screenshot, calibration, backend="template")
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
        selected = read_selected_unit(screenshot, calibration)
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
    if screenshot is None:
        if rect is None:
            return tuple(checks)
        screenshot, _width, _height = capture_screenshot()
    checks.extend(inspect_capture(profile, screenshot))
    checks.append(
        PreflightCheck(
            "capture_to_input_geometry",
            window_ok,
            "capture origin and input window offset agree only when the exact game window is found",
        )
    )
    return tuple(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only supported-profile preflight")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(namespace=_PreflightArgs())
    profile = load_profile(args.profile)
    screenshot = args.screenshot.read_bytes() if args.screenshot is not None else None
    checks = run_preflight(profile, screenshot)
    print(json.dumps([asdict(check) for check in checks], indent=2))
    raise SystemExit(0 if all(check.passed for check in checks) else 1)


if __name__ == "__main__":
    main()
