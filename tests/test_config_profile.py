"""Tests for arena/config_profile.py (Phase 6.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from arena.config_profile import ConfigProfile, RaceConfig

_PROFILES_DIR = Path(__file__).parent.parent / "arena" / "profiles"


def test_race_config_from_yaml_loads_profiles() -> None:
    config = RaceConfig.from_yaml(_PROFILES_DIR / "v1.yaml")
    assert len(config.profiles) == 2


def test_config_profile_has_correct_name() -> None:
    profile = ConfigProfile(name="my-variant")
    assert profile.name == "my-variant"


def test_config_profile_rejects_temperature_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ConfigProfile(name="bad", temperature=1.5)


def test_race_config_requires_at_least_two_profiles() -> None:
    with pytest.raises(ValidationError):
        RaceConfig(turns=10, profiles=[ConfigProfile(name="only-one")])
