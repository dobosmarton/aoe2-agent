"""Tests for arena/scenarios.py — default-scenarios library sanity checks."""

from __future__ import annotations

import pytest
from arena.scenarios import DEFAULT_SCENARIOS, get_scenario


def test_default_scenarios_has_four_entries() -> None:
    assert len(DEFAULT_SCENARIOS) == 4


def test_default_scenarios_names_are_unique() -> None:
    names = [s.name for s in DEFAULT_SCENARIOS]
    assert len(names) == len(set(names))


def test_get_scenario_returns_named_entry() -> None:
    assert get_scenario("balanced").initial_state.food == 200.0


def test_get_scenario_raises_on_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_scenario("nonexistent")
