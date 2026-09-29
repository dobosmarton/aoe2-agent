"""The 4v4 profile cannot infer team identity from player color alone."""

import json
from pathlib import Path

import pytest
from gameplay_agent.game_profile import GameProfile, load_profile, recording_qualification
from gameplay_agent.preflight import inspect_capture, run_preflight
from pydantic import ValidationError

_EXAMPLE = Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.example.json"
_HIGHLAND = Path(__file__).parents[1] / "apps/agent/profiles/magyars_highland_4v4.example.json"


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
