"""Unit tests for A2 reflective trace capture + reflective proposer.

Trace helpers are pure. The reflective-prompt test mocks the anthropic client
so it asserts the prompt content without any network call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from autoresearch.metrics import GameScore
from autoresearch.trace import (
    GameTrace,
    TurnTrace,
    build_game_trace,
    format_trace_excerpt,
    load_recent_traces,
    save_trace,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _score(**over: float) -> GameScore:
    base = {"age": 0.0, "age_speed": 0.0, "economy": 0.1, "action_success": 0.4, "survival": 0.5}
    base.update(over)
    return GameScore(
        composite=0.3,
        age=base["age"],
        age_speed=base["age_speed"],
        economy=base["economy"],
        action_success=base["action_success"],
        survival=base["survival"],
        raw_metrics={"game_end_reason": "timeout"},
    )


# ---------------------------------------------------------------------------
# build_game_trace / format_trace_excerpt
# ---------------------------------------------------------------------------


def test_build_game_trace_from_memory() -> None:
    from gameplay_agent.memory import AgentMemory

    mem = AgentMemory()
    mem.create_turn(
        reasoning="gather food", actions=[{"type": "press", "key": "q"}], observations={}
    )
    trace = build_game_trace(mem, _score())
    assert len(trace.turns) == 1
    assert trace.turns[0].reasoning == "gather food"
    assert trace.components["survival"] == 0.5
    assert trace.end_reason == "timeout"


def test_format_trace_excerpt_includes_components_and_turn() -> None:
    trace = GameTrace(
        turns=[TurnTrace(1, "build house", "build(q)", "- CONFIRMED built: house")],
        components={
            "survival": 0.5,
            "population": 0.2,
            "age": 0.0,
            "economy": 0.1,
            "action_success": 0.4,
        },
        composite=0.3,
        end_reason="timeout",
    )
    out = format_trace_excerpt(trace)
    assert "survival=0.50" in out
    assert "build house" in out
    assert "CONFIRMED" in out


def test_save_and_load_recent_traces(tmp_path: Path) -> None:
    trace = GameTrace(
        turns=[TurnTrace(1, "x", "y", "")],
        components={"survival": 0.5},
        composite=0.3,
        end_reason="timeout",
    )
    save_trace(trace, "exp_1", traces_dir=tmp_path)
    loaded = load_recent_traces(5, traces_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].composite == 0.3
    assert loaded[0].turns[0].reasoning == "x"


# ---------------------------------------------------------------------------
# propose_changes reflective prompt construction (mocked client)
# ---------------------------------------------------------------------------


def test_propose_changes_builds_reflective_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoresearch.prompt_mutator as pm

    monkeypatch.setattr(pm, "make_text_completer", lambda *_a, **_k: object())
    mutator = pm.PromptMutator()

    captured: dict[str, object] = {}

    def _fake_complete(system: str, user: str, max_tokens: int) -> str:
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        return '[{"description": "d", "old_text": "o", "new_text": "n", "rationale": "r"}]'

    mutator.completer = SimpleNamespace(complete=_fake_complete)

    trace = GameTrace(
        turns=[TurnTrace(1, "queued villagers", "press(q)", "")],
        components={
            "survival": 0.5,
            "population": 0.2,
            "age": 0.0,
            "economy": 0.1,
            "action_success": 0.4,
        },
        composite=0.3,
        end_reason="timeout",
    )
    out = mutator.propose_changes("PROMPT", [trace], {"survival": 0.5, "economy": 0.1}, [], n=2)

    assert len(out) == 1
    user_text = str(captured["user"])
    assert "queued villagers" in user_text  # trace reasoning present
    assert "survival" in user_text  # component breakdown present
    assert captured["system"] == pm.REFLECTIVE_MUTATOR_SYSTEM
