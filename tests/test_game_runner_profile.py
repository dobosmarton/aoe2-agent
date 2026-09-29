"""Recorded games require a qualified roster and bindings, not map verification."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from autoresearch import game_runner
from gameplay_agent.game_profile import load_profile

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_HIGHLAND = Path(__file__).parents[1] / "apps/agent/profiles/magyars_highland_4v4.example.json"


def _without_game_input(monkeypatch: MonkeyPatch) -> None:
    async def game_loop(**kwargs: object) -> object:
        return kwargs["memory"]

    monkeypatch.setattr(game_runner, "game_loop", game_loop)
    monkeypatch.setattr(game_runner, "ExecutorProvider", object)
    monkeypatch.setattr(game_runner, "TypeSafePolicyAdvisor", lambda **_kwargs: object())


def test_recorded_game_without_profile_is_not_score_valid(monkeypatch: MonkeyPatch) -> None:
    _without_game_input(monkeypatch)
    monkeypatch.setattr(game_runner.config, "game_profile_path", None)

    result = asyncio.run(game_runner.run_game(extract_memories=False))

    assert result["metrics"]["score_valid"] is False


def test_qualified_highland_profile_allows_recorded_score(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    _without_game_input(monkeypatch)
    profile = load_profile(_HIGHLAND).model_copy(
        update={
            "roster_verified": True,
            "hotkeys_verified": True,
            "ownership_verified": True,
        }
    )
    path = tmp_path / "highland.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(game_runner.config, "game_profile_path", path)

    result = asyncio.run(game_runner.run_game(extract_memories=False))

    assert result["metrics"]["score_valid"] is True
