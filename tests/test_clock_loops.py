"""The supervisor: it starts the clocks, and it labels the end.

Everything here runs on `FakeSource` / `FakeActuator`, so there is no game, no
detector and no LLM. That is the point of the environment seam.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from gameplay_agent import executor as ex
from gameplay_agent import game_loop as gl
from gameplay_agent.loops import deliberate, perceive
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.providers.base import LLMResult
from gameplay_agent.providers.executor_provider import ExecutorProvider
from gameplay_agent.providers.strategist import StrategistProvider

from tests.factories import make_entity as _ent
from tests.loop_fakes import FakeActuator, FakeSource


class _FakeProvider(ExecutorProvider):
    """No LLM runs in the rules-only supervisor; only its wire is warmed."""


@pytest.fixture
def loop_seams(monkeypatch, tmp_path):
    """A supervisor with no game behind it. Returns the source and actuator."""
    source = FakeSource([Perception(entities=(_ent("town_center", (0.0, 0.0)),))])
    actuator = FakeActuator()
    monkeypatch.setattr(gl, "GameSource", lambda **_kwargs: source)
    monkeypatch.setattr(gl, "GameActuator", lambda: actuator)
    monkeypatch.setattr(gl, "_init_detector", lambda: None)
    monkeypatch.setattr(gl, "_init_frame_differ", lambda: None)
    monkeypatch.setattr(gl, "warm_up_ocr", lambda: None)
    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: True)
    monkeypatch.setattr(perceive.config, "perceive_interval", 0.0)
    monkeypatch.setattr(gl.config, "log_dir", tmp_path)
    monkeypatch.setattr(gl.config, "save_screenshots", False)
    # No network: the strategist would otherwise try a real call per test.
    monkeypatch.setattr(StrategistProvider, "should_run", lambda *_a, **_k: False)
    ex.reset_build_gates()
    yield source, actuator
    ex.reset_build_gates()


def _play(frames: int = 2) -> AgentMemory:
    """Run one game bounded by a frame budget."""
    return asyncio.run(gl.game_loop(_FakeProvider(), max_iterations=frames))


# ---------------------------------------------------------------------------
# The clocks run
# ---------------------------------------------------------------------------


def test_the_supervisor_runs_on_fakes(loop_seams) -> None:
    """No game, no LLM — the seam is what makes this testable at all."""
    source, _actuator = loop_seams
    _play(frames=3)
    assert source.captures == 3


def test_the_run_reports_the_clocks_architecture(loop_seams) -> None:
    """`loop_arch` flips on the presence of an act tick, and the ledger reads it."""
    memory = _play()
    assert memory.get_metrics_snapshot()["loop_arch"] == "clocks"


def test_the_clock_starts_without_an_llm_turn(loop_seams) -> None:
    """`game_start_time` used to be set only inside `create_turn`. Unstarted, it
    zeroes `survival_time` and every `age_times` entry — 0.30 of the score."""
    memory = _play()
    assert memory.get_metrics_snapshot()["survival_time"] > 0.0


def test_the_opening_commands_run_before_the_first_frame(loop_seams) -> None:
    """The scout explores during the slowest perception pass of the game."""
    _source, actuator = loop_seams
    _play()
    assert [a["type"] for a in actuator.batches[0]] == ["scroll", "press", "press"]


# ---------------------------------------------------------------------------
# How a game ends
# ---------------------------------------------------------------------------


def test_the_frame_budget_ends_the_run(loop_seams) -> None:
    assert _play(frames=2).game_end_reason == "iterations_exhausted"


def test_the_frame_budget_is_counted_in_frames(loop_seams) -> None:
    """An iteration is a perceive frame now, not an LLM turn."""
    source, _actuator = loop_seams
    _play(frames=2)
    assert source.captures == 2


def test_the_time_budget_ends_the_run(loop_seams) -> None:
    memory = asyncio.run(gl.game_loop(_FakeProvider(), time_budget=0.01))
    assert memory.game_end_reason == "timeout"


def test_a_missing_window_ends_the_run(loop_seams, monkeypatch) -> None:
    monkeypatch.setattr(perceive, "is_game_running", lambda: False)
    assert _play().game_end_reason == "game_not_found"


def test_a_source_failure_still_labels_the_run(loop_seams, monkeypatch) -> None:
    """T-543: a stop that bypasses both except clauses (CancelledError is a
    BaseException) once logged an empty `game_end_reason`."""
    source, _actuator = loop_seams

    async def _cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(source, "capture", _cancelled)
    memory = AgentMemory()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(gl.game_loop(_FakeProvider(), max_iterations=1, memory=memory))
    assert memory.game_end_reason == "interrupted"


def test_the_source_is_closed(loop_seams) -> None:
    """The overlay lives behind the source, so the run must release it."""
    source, _actuator = loop_seams
    _play()
    assert source.closed is True


# ---------------------------------------------------------------------------
# The rescan hook — registered over the inline one
# ---------------------------------------------------------------------------


def test_the_refresh_hook_replaces_the_inline_rescan(loop_seams) -> None:
    """Composite handlers rescan from inside `execute_action`. Pointing the hook
    at the frame pipe is what keeps a detection off the act task."""
    _play()
    assert ex.get_rescan_fn() is not None


# ---------------------------------------------------------------------------
# The whole phase in one test: a slow LLM must not stop the agent acting
# ---------------------------------------------------------------------------


async def _stalled_plan(*_args: object, **_kwargs: object) -> LLMResult:
    """An LLM that never answers inside a game's lifetime."""
    await asyncio.sleep(30)
    return LLMResult(reasoning="too late", actions=[], observations={})


def test_act_ticks_while_the_llm_stalls(loop_seams, monkeypatch) -> None:
    """The load-bearing check. Under the old turn loop a 30 s call stopped the
    agent dead; here the act loop keeps deciding on every frame."""
    monkeypatch.setattr(deliberate.config, "deliberate_interval", 1)  # ask every frame
    monkeypatch.setattr(ExecutorProvider, "plan", _stalled_plan)
    memory = _play(frames=5)
    assert memory.get_metrics_snapshot()["loop_arch"] == "clocks"


def test_the_frames_keep_coming_while_the_llm_stalls(loop_seams, monkeypatch) -> None:
    source, _actuator = loop_seams

    monkeypatch.setattr(deliberate.config, "deliberate_interval", 1)
    monkeypatch.setattr(ExecutorProvider, "plan", _stalled_plan)
    _play(frames=5)
    assert source.captures == 5


# ---------------------------------------------------------------------------
# Warm-ups — deferred SDK imports must not land on the event loop
# ---------------------------------------------------------------------------


def test_the_wire_is_warmed_once(loop_seams, monkeypatch) -> None:
    """Pay the 115-module import once, off the loop, not inside a frame."""
    provider = _FakeProvider()
    warmed: list[bool] = []
    monkeypatch.setattr(provider.wire, "warm_up", lambda: warmed.append(True))
    asyncio.run(gl.game_loop(provider, max_iterations=1))
    assert warmed == [True]


def test_a_failing_wire_warm_up_does_not_stop_the_game(loop_seams, monkeypatch) -> None:
    """Warm-up is opportunistic, exactly as `warm_up_ocr` is."""
    provider = _FakeProvider()

    def _boom() -> None:
        raise RuntimeError("no SDK on this host")

    monkeypatch.setattr(provider.wire, "warm_up", _boom)
    memory = asyncio.run(gl.game_loop(provider, max_iterations=1))
    assert memory.game_end_reason != "error"


def test_a_pending_ocr_warm_up_does_not_leak_on_exit(loop_seams, monkeypatch) -> None:
    """CI-only flake: cancelling a still-running warm-up must not raise, and
    `CancelledError` is a BaseException that `suppress(Exception)` misses."""
    monkeypatch.setattr(gl.config, "ocr_backend", "rapidocr")
    monkeypatch.setattr(gl, "warm_up_ocr", lambda: time.sleep(0.3))
    assert _play(frames=1).game_end_reason == "iterations_exhausted"
