"""Tests for arena/metrics.py (Phase 6). Offline — uses build_mock_invoke."""

from __future__ import annotations

import asyncio

from arena.config_profile import ConfigProfile, RaceConfig
from arena.metrics import VariantMetrics, extract_metrics, summarise
from arena.race import VariantResult, race_with_mock
from evaluation.world_sim import WorldState


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


def _run_results() -> list[VariantResult]:
    return asyncio.run(race_with_mock(_two_profile_config(), _initial_state()))


def test_extract_metrics_reads_final_state_from_last_turn() -> None:
    results = _run_results()
    metrics = extract_metrics(results[0])
    assert isinstance(metrics, VariantMetrics)
    assert metrics.turns_completed == 3


def test_extract_metrics_total_cost_from_loop_result() -> None:
    results = _run_results()
    metrics = extract_metrics(results[0])
    assert metrics.total_cost_usd == results[0].loop_result.total_cost_usd


def test_summarise_includes_all_profile_names() -> None:
    results = _run_results()
    table = summarise(results)
    assert "profile-a" in table
    assert "profile-b" in table


def test_summarise_ranks_by_population_descending() -> None:
    results = _run_results()
    all_metrics = sorted(
        [extract_metrics(r) for r in results], key=lambda m: m.final_pop, reverse=True
    )
    table = summarise(results)
    positions = [table.index(m.name) for m in all_metrics]
    assert positions == sorted(positions)
