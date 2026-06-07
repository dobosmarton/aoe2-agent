"""Tests for the autoresearch successive-halving tournament (A1) and the
JSON-array extraction it relies on.

All external effects — games, git, the mutator LLM, and the TSV ledger — are
stubbed so the suite runs offline (no network, no real git, no human input).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import autoresearch.orchestrator as orch_module
from autoresearch.json_utils import extract_json_array
from autoresearch.metrics import GameScore

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import pytest


def _run(coro: Awaitable[object]) -> object:
    return asyncio.run(coro)


def _score(composite: float) -> GameScore:
    return GameScore(
        composite=composite,
        survival=0.0,
        population=0.0,
        age=0.0,
        economy=0.0,
        action_success=0.0,
        raw_metrics={},
    )


# ---------------------------------------------------------------------------
# extract_json_array
# ---------------------------------------------------------------------------


def test_extract_json_array_parses_plain_list() -> None:
    assert extract_json_array('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_extract_json_array_drops_non_dict_elements() -> None:
    assert extract_json_array('[{"a": 1}, 7, "x"]') == [{"a": 1}]


def test_extract_json_array_wraps_bare_object() -> None:
    assert extract_json_array('{"a": 1}') == [{"a": 1}]


def test_extract_json_array_from_code_block() -> None:
    assert extract_json_array('prose\n```json\n[{"a": 1}]\n```\ntrailing') == [{"a": 1}]


def test_extract_json_array_returns_empty_on_garbage() -> None:
    assert extract_json_array("no json here") == []


# ---------------------------------------------------------------------------
# PromptMutator candidate parsing
# ---------------------------------------------------------------------------


def test_parse_changes_keeps_only_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoresearch.prompt_mutator as pm

    monkeypatch.setattr(pm.anthropic, "Anthropic", lambda *_a, **_k: object())
    mutator = pm.PromptMutator()
    text = (
        '[{"description": "d", "old_text": "o", "new_text": "n", "rationale": "r"},'
        ' {"description": "missing fields"}]'
    )
    changes = mutator._parse_changes(text)
    assert len(changes) == 1
    assert changes[0]["old_text"] == "o"


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


class _FakeMutator:
    """Stand-in mutator: no network; applies always succeed."""

    def __init__(self, changes: list[dict]) -> None:
        self._changes = changes
        self.applied: list[tuple[str, str]] = []

    def read_current_prompt(self) -> str:
        return "PROMPT a b c"

    def propose_changes(self, *_a: object, n: int = 3, **_k: object) -> list[dict]:
        return self._changes[:n]

    def apply_change(self, old_text: str, new_text: str) -> bool:
        self.applied.append((old_text, new_text))
        return True


def _make_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    changes: list[dict],
    composites: list[float],
    best_score: float = 0.0,
) -> tuple[orch_module.Orchestrator, list[dict], dict[str, int]]:
    monkeypatch.setattr(orch_module, "get_best_score", lambda loop=None: best_score)
    monkeypatch.setattr(orch_module, "PromptMutator", lambda: _FakeMutator(changes))
    monkeypatch.setattr(orch_module, "get_next_experiment_id", lambda: "exp_test")
    monkeypatch.setattr(orch_module, "get_recent_experiments", lambda n=5: [])
    logged: list[dict] = []
    monkeypatch.setattr(orch_module, "log_experiment", lambda **kw: logged.append(kw))
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    scores = iter(composites)

    async def _fake_run_game(**_kw: object) -> dict:
        return {"score": _score(next(scores))}

    monkeypatch.setattr(orch_module, "run_game", _fake_run_game)

    orch = orch_module.Orchestrator()
    counters = {"reverts": 0}

    def _count_revert() -> None:
        counters["reverts"] += 1

    monkeypatch.setattr(orch, "git_commit", lambda _msg: "sha")
    monkeypatch.setattr(orch, "git_revert_prompt", _count_revert)
    return orch, logged, counters


def _changes() -> list[dict]:
    return [
        {"description": "A", "old_text": "a", "new_text": "A", "rationale": ""},
        {"description": "B", "old_text": "b", "new_text": "B", "rationale": ""},
        {"description": "C", "old_text": "c", "new_text": "C", "rationale": ""},
    ]


def test_tournament_keeps_best_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    orch, _logged, _counters = _make_orchestrator(monkeypatch, _changes(), [0.1, 0.9, 0.3])
    result = _run(orch.run_tournament(n_candidates=3, halving_rounds=1, keep_fraction=0.5))
    assert result["accepted"] is True
    assert result["description"] == "B"  # the 0.9 candidate


def test_tournament_reverts_after_each_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    orch, _logged, counters = _make_orchestrator(monkeypatch, _changes(), [0.1, 0.9, 0.3])
    _run(orch.run_tournament(n_candidates=3, halving_rounds=1, keep_fraction=0.5))
    assert counters["reverts"] == 3  # one revert per trial game; winner re-applied after


def test_tournament_rejects_winner_below_epsilon(monkeypatch: pytest.MonkeyPatch) -> None:
    orch, _logged, _counters = _make_orchestrator(
        monkeypatch, _changes(), [0.1, 0.9, 0.3], best_score=0.95
    )
    result = _run(orch.run_tournament(n_candidates=3, halving_rounds=1, keep_fraction=0.5))
    assert result["accepted"] is False  # 0.9 < 0.95 - 0.02


def test_tournament_skips_inapplicable_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    changes = [
        {"description": "A", "old_text": "a", "new_text": "A", "rationale": ""},
        {"description": "Z", "old_text": "ZZZ_absent", "new_text": "Z", "rationale": ""},
    ]
    orch, _logged, counters = _make_orchestrator(monkeypatch, changes, [0.5])
    result = _run(orch.run_tournament(n_candidates=2, halving_rounds=1, keep_fraction=0.5))
    assert result["candidates"] == 2  # both proposed
    assert result["description"] == "A"  # only A applies to the prompt
    assert counters["reverts"] == 1  # only one trial game ran


def test_tournament_logs_each_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    orch, logged, _counters = _make_orchestrator(monkeypatch, _changes(), [0.1, 0.9, 0.3])
    _run(orch.run_tournament(n_candidates=3, halving_rounds=1, keep_fraction=0.5))
    trial_rows = [r for r in logged if r.get("round_num") == "1"]
    assert len(trial_rows) == 3
    assert all(r["accepted"] is False for r in trial_rows)
    assert {r["candidate_id"] for r in trial_rows} == {"c1", "c2", "c3"}


# ---------------------------------------------------------------------------
# A2 — Pareto retention in _keep_top
# ---------------------------------------------------------------------------


def _game(
    composite: float,
    survival: float = 0.0,
    population: float = 0.0,
    age: float = 0.0,
    economy: float = 0.0,
    action_success: float = 0.0,
) -> GameScore:
    return GameScore(
        composite=composite,
        survival=survival,
        population=population,
        age=age,
        economy=economy,
        action_success=action_success,
        raw_metrics={},
    )


def _cand(cid: str, game: GameScore) -> orch_module._Candidate:
    return orch_module._Candidate(
        candidate_id=cid,
        change={"description": cid, "old_text": "o", "new_text": "n"},
        games=[game],
    )


def test_keep_top_retains_non_dominated() -> None:
    best = _cand("c_best", _game(0.8, 0.8, 0.8, 0.8, 0.8, 0.8))
    offaxis = _cand("c_offaxis", _game(0.3, age=1.0))  # weak composite, best on one axis
    weak = _cand("c_weak", _game(0.4, 0.5, 0.5, 0.5, 0.5, 0.5))  # dominated by c_best
    kept = orch_module.Orchestrator._keep_top([best, offaxis, weak], keep_fraction=0.5)
    ids = {c.candidate_id for c in kept}
    assert "c_best" in ids  # top by composite
    assert "c_offaxis" in ids  # non-dominated off-axis survivor retained
    assert "c_weak" not in ids  # dominated → dropped
