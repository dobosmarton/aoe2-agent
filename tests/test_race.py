"""Tests for arena/race.py (Phase 6). Offline — uses build_mock_invoke."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from arena.config_profile import ConfigProfile, RaceConfig
from arena.race import VariantResult, race_with_mock
from evaluation.world_sim import WorldState

if TYPE_CHECKING:
    from evaluation.event_log import Event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state() -> WorldState:
    return WorldState(
        food=200.0,
        wood=150.0,
        gold=0.0,
        stone=0.0,
        population=8,
        pop_cap=25,
        age="Dark Age",
        buildings=[],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def _two_profile_config(turns: int = 3) -> RaceConfig:
    return RaceConfig(
        turns=turns,
        profiles=[
            ConfigProfile(name="profile-a"),
            ConfigProfile(name="profile-b"),
        ],
    )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _run(config: RaceConfig, **kwargs: object) -> list[VariantResult]:
    return asyncio.run(race_with_mock(config, _initial_state(), **kwargs))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_race_returns_one_result_per_profile() -> None:
    results = _run(_two_profile_config())
    assert len(results) == 2


def test_race_result_run_ids_are_distinct() -> None:
    results = _run(_two_profile_config())
    run_ids = {r.loop_result.run_id for r in results}
    assert len(run_ids) == 2


def test_race_each_result_carries_its_profile_name() -> None:
    results = _run(_two_profile_config())
    names = {r.profile.name for r in results}
    assert names == {"profile-a", "profile-b"}


def test_race_emits_events_tagged_with_correct_run_ids() -> None:
    sink = _RecordingSink()
    results = asyncio.run(race_with_mock(_two_profile_config(turns=1), _initial_state(), sink=sink))
    expected = {r.loop_result.run_id for r in results}
    observed = {e.run_id for e in sink.events}
    assert observed == expected
