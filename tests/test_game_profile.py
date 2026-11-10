"""The 4v4 profile cannot infer team identity from player color alone."""

import json
from pathlib import Path

import pytest
from gameplay_agent.game_profile import (
    GameProfile,
    load_profile,
    platform_problem,
    recording_qualification,
)
from gameplay_agent.preflight import inspect_capture, run_preflight
from gameplay_agent.window import Platform
from pydantic import ValidationError

_EXAMPLE = Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.example.json"
_HIGHLAND = Path(__file__).parents[1] / "apps/agent/profiles/magyars_highland_4v4.example.json"
_MAC = Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.macos.example.json"


def test_example_is_structurally_valid_but_not_qualified() -> None:
    profile = load_profile(_EXAMPLE)
    assert len(profile.players) == 8
    assert profile.roster_verified is False
    assert profile.hotkeys_verified is False
    assert profile.ownership_verified is False


def test_highland_is_a_supported_declared_map() -> None:
    data = load_profile(_EXAMPLE).model_dump()
    data["map_name"] = "Highland"

    assert GameProfile.model_validate(data).map_name == "Highland"
    assert load_profile(_HIGHLAND).map_name == "Highland"


def test_recorded_profile_requires_a_verified_roster_but_not_map() -> None:
    profile = load_profile(_EXAMPLE)

    assert "map" not in recording_qualification(profile)
    assert "roster" in recording_qualification(profile)
    assert recording_qualification(None) == ("profile",)


def test_profile_rejects_a_team_relationship_mismatch() -> None:
    data = load_profile(_EXAMPLE).model_dump()
    data["players"][1]["team"] = 2
    with pytest.raises(ValidationError, match="relationships contradict"):
        GameProfile.model_validate(data)


def test_profile_rejects_wrong_civilization_or_player_count() -> None:
    data = load_profile(_EXAMPLE).model_dump()
    data["civilization"] = "Britons"
    with pytest.raises(ValidationError):
        GameProfile.model_validate(data)
    data["civilization"] = "Magyars"
    data["players"] = data["players"][:-1]
    with pytest.raises(ValidationError, match="eight player slots"):
        GameProfile.model_validate(data)


def test_profile_rejects_unknown_fields_and_coerced_values() -> None:
    data = load_profile(_EXAMPLE).model_dump()
    data["unexpected"] = "unqualified assumption"
    with pytest.raises(ValidationError, match="unexpected"):
        GameProfile.model_validate(data)
    del data["unexpected"]
    data["capture_width"] = "3024"
    with pytest.raises(ValidationError, match="capture_width"):
        GameProfile.model_validate(data)
    with pytest.raises(ValidationError, match="capture_width"):
        GameProfile.model_validate_json(json.dumps(data))


def test_preflight_reads_calibrated_hud_but_does_not_self_certify_window(monkeypatch) -> None:
    from gameplay_agent import preflight

    screenshot = (
        Path(__file__).parents[1] / "apps/agent/src/vision_fixtures/real_1672_dark_midgame.jpg"
    ).read_bytes()
    profile = load_profile(_EXAMPLE)
    capture_checks = {check.name: check for check in inspect_capture(profile, screenshot)}
    assert capture_checks["capture_geometry"].passed
    assert capture_checks["hud_reading"].passed
    monkeypatch.setattr(preflight, "get_game_window_rect", lambda: None)
    checks = {check.name: check for check in run_preflight(profile, screenshot)}
    assert not checks["window"].passed
    assert not checks["roster"].passed
    assert not checks["hotkeys"].passed
    assert not checks["ownership_colors"].passed
    assert "map" not in checks


_MAC = Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.macos.example.json"


def test_the_mac_example_is_the_full_screen_game_area() -> None:
    profile = load_profile(_MAC)
    assert profile.platform == "macos"
    assert (profile.capture_width, profile.capture_height) == (3024, 1898)
    assert load_profile(_EXAMPLE).platform == "windows"  # Older profiles default to Windows.


@pytest.mark.parametrize(
    ("source", "change"),
    [
        (_EXAMPLE, {"capture_height": 1898}),  # A Windows profile with the Mac's height.
        (_MAC, {"capture_height": 1672}),  # And the reverse.
        (_EXAMPLE, {"zoom_steps": 4}),  # Mac-only settings on Windows.
        (_EXAMPLE, {"modifier_map": {"ctrl": "command"}}),
        (_MAC, {"zoom_steps": 31}),
        (_MAC, {"modifier_map": {"ctrl": "super"}}),
    ],
)
def test_platform_settings_must_agree(source: Path, change: dict[str, object]) -> None:
    data = {**json.loads(source.read_text()), **change}
    with pytest.raises(ValidationError):
        GameProfile.model_validate_json(json.dumps(data))


def test_a_mac_profile_takes_a_zoom_and_a_modifier_map() -> None:
    data = {**json.loads(_MAC.read_text()), "zoom_steps": 6, "modifier_map": {"ctrl": "command"}}
    profile = GameProfile.model_validate_json(json.dumps(data))
    assert profile.zoom_steps == 6
    assert profile.modifier_map == {"ctrl": "command"}


def test_the_mac_game_needs_a_profile() -> None:
    """Without one, games 3 and 4 ordered with the right button and every order failed."""
    assert platform_problem(None, "macos") is not None


def test_the_mac_game_refuses_a_windows_profile() -> None:
    assert platform_problem(load_profile(_EXAMPLE), "macos") is not None


def test_a_mac_profile_is_refused_on_windows() -> None:
    assert platform_problem(load_profile(_MAC), "windows") is not None


@pytest.mark.parametrize(
    ("profile_path", "platform"),
    [(_MAC, "macos"), (_EXAMPLE, "windows"), (None, "windows"), (None, "none")],
    ids=["mac", "windows", "windows-without-profile", "no-window"],
)
def test_a_matching_or_optional_profile_is_accepted(
    profile_path: Path | None, platform: Platform
) -> None:
    profile = load_profile(profile_path) if profile_path is not None else None
    assert platform_problem(profile, platform) is None
