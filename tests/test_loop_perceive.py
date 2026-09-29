"""Unit tests for loops/perceive.py — the clock that reads the world."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops import perceive
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.snapshot import Perception, SpatialRefresh
from gameplay_agent.loops.source import GameSource, Sighting, frame_refresh
from gameplay_agent.memory import AgentMemory
from gameplay_agent.turn_timing import PERCEIVE_LOOP, TickTimings

from tests.factories import make_entity as _ent
from tests.loop_fakes import FakeActuator, FakeSource

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


@pytest.fixture
def gates():
    ex.reset_build_gates()
    yield
    ex.reset_build_gates()


def _context(tmp_path, source: FakeSource) -> LoopContext:
    return LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=source,
        actuator=FakeActuator(),
    )


def test_a_pass_publishes_a_frame(tmp_path, gates) -> None:
    ctx = _context(tmp_path, FakeSource([Perception(width=800, height=600)]))
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.width == 800


def test_spatial_refresh_detaches_mutable_hud_readings() -> None:
    source = {"food": 150}
    refresh = SpatialRefresh(time.monotonic(), 0, True, source)
    source["food"] = 0
    assert refresh.hud_readings["food"] == 150
    with pytest.raises(TypeError):
        refresh.hud_readings["food"] = 20


def test_perception_normalizes_mutable_entities_once() -> None:
    original = {
        "id": "sheep-1",
        "class": "sheep",
        "center": [120, 130],
        "bbox": [110, 120, 130, 140],
        "confidence": 0.95,
    }
    frame = Perception(entities=(original,))
    original["center"] = [400, 400]
    assert frame.entities[0].center == (120, 130)
    assert frame.entities[0].id == "sheep-1"
    assert Perception(entities=frame.entities).entities == frame.entities


def test_sighting_detaches_mutable_ownership() -> None:
    from detection.inference.ownership import Owner

    source = {"unit-1": (Owner.OWN, 0.9)}
    sighting = Sighting(Perception(), source)
    source.clear()
    assert sighting.ownership["unit-1"] == (Owner.OWN, 0.9)
    with pytest.raises(TypeError):
        sighting.ownership["unit-2"] = (Owner.ENEMY, 0.9)


def test_the_frame_id_is_the_pass_number(tmp_path, gates) -> None:
    """`act_decided` names the frame it acted on, so the id must be the pipe's."""
    ctx = _context(tmp_path, FakeSource([Perception(), Perception()]))
    _run(perceive.perceive_once(ctx, tick=1))
    _run(perceive.perceive_once(ctx, tick=2))
    frame = ctx.frames.latest()
    assert frame is not None and frame.tick == 2


def test_a_pass_records_its_own_latency(tmp_path, gates) -> None:
    """`loop_arch` and the perceive budget both read this recorder."""
    ctx = _context(tmp_path, FakeSource())
    _run(perceive.perceive_once(ctx, tick=1))
    assert PERCEIVE_LOOP in ctx.latency.snapshot().loops


def test_spatial_capture_includes_hud_baseline(monkeypatch) -> None:
    game_source = GameSource()
    detected: list[bytes] = []

    async def screen(tick, timings):
        assert tick is None
        return b"jpeg", b"native-hud", 800, 600, time.monotonic()

    async def detect(screenshot, *, fresh=False):
        detected.append(screenshot)
        assert fresh
        return []

    async def hud(*_args):
        return {"food": 200}

    monkeypatch.setattr(game_source, "_screen", screen)
    monkeypatch.setattr(game_source, "_detect_entities", detect)
    monkeypatch.setattr(game_source, "_hud", hud)

    refresh = _run(game_source.capture_spatial(TickTimings()))

    assert refresh.spatial_valid is True
    assert detected == [b"jpeg"]
    assert refresh.hud_readings == {"food": 200}


def test_selection_refresh_skips_object_detection(monkeypatch) -> None:
    source = GameSource()

    async def screen(_tick, _timings):
        return b"jpeg", b"native-hud", 3024, 1672, time.monotonic()

    async def detect(_screenshot, *, fresh=False):
        pytest.fail("selection/HUD verification must not wait for object detection")

    async def hud(*_args):
        return {"food": 150}

    monkeypatch.setattr(source, "_screen", screen)
    monkeypatch.setattr(source, "_detect_entities", detect)
    monkeypatch.setattr(source, "_hud", hud)
    monkeypatch.setattr(
        "gameplay_agent.loops.source.read_selected_unit",
        lambda _screenshot, _calibration: "town_center",
    )

    refresh = _run(source.capture_spatial(TickTimings(), selection_only=True))
    assert refresh.spatial_valid
    assert refresh.selected_unit == "town_center"
    assert refresh.hud_readings == {"food": 150}


def test_spatial_capture_rejects_input_that_changes_during_detection(monkeypatch) -> None:
    ledger = ex.ActionLedger()
    game_source = GameSource(ledger=ledger)

    async def screen(_tick, _timings):
        return b"jpeg", b"native-hud", 800, 600, time.monotonic()

    async def detect(_screenshot, *, fresh=False):
        assert fresh
        ledger.note_input()
        return []

    monkeypatch.setattr(game_source, "_screen", screen)
    monkeypatch.setattr(game_source, "_detect_entities", detect)

    refresh = _run(game_source.capture_spatial(TickTimings()))

    assert refresh.spatial_valid is False


def test_requested_refresh_runs_hud_and_detection_on_same_capture_concurrently(
    monkeypatch,
) -> None:
    source = GameSource()
    hud_started = asyncio.Event()
    detect_started = asyncio.Event()

    async def screen(_tick, _timings):
        return b"captured-frame", b"native-hud", 800, 600, time.monotonic()

    async def hud(screenshot, _tick, _timings, full_size):
        assert screenshot == b"native-hud"
        assert full_size == (800, 600)
        hud_started.set()
        await detect_started.wait()
        return {"food": 200}

    async def detect(screenshot, *, fresh=False):
        assert screenshot == b"captured-frame"
        assert fresh
        detect_started.set()
        await hud_started.wait()
        return []

    monkeypatch.setattr(source, "_screen", screen)
    monkeypatch.setattr(source, "_hud", hud)
    monkeypatch.setattr(source, "_detect_entities", detect)
    refresh = _run(asyncio.wait_for(source.capture_spatial(TickTimings()), timeout=0.5))
    assert refresh.hud_readings == {"food": 200}


def test_requested_refresh_preempts_an_unfinished_background_detection(
    tmp_path, monkeypatch
) -> None:
    class SlowSource:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def capture(self, _tick, _timings):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

        async def capture_hud(self, _tick, _timings):
            return None

        async def capture_spatial(self, _timings, *, selection_only=False):
            return SpatialRefresh(time.monotonic(), 0, True)

        def close(self):
            pass

    source = SlowSource()
    ctx = _context(tmp_path, source)

    async def playable(_ctx):
        return True

    monkeypatch.setattr(perceive, "_wait_until_playable", playable)

    async def drive() -> None:
        loop = asyncio.create_task(perceive.perceive_loop(ctx))
        await asyncio.wait_for(source.started.wait(), timeout=1.0)
        assert await asyncio.wait_for(frame_refresh(ctx.frames)(), timeout=1.0)
        ctx.request_stop("test_complete")
        await asyncio.wait_for(loop, timeout=1.0)
        assert source.cancelled.is_set()

    asyncio.run(drive())


def test_selection_refresh_request_reaches_perception_without_detection(
    tmp_path, monkeypatch
) -> None:
    class SelectionSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.selection_only: bool | None = None

        async def capture_spatial(self, timings, *, selection_only=False):
            self.selection_only = selection_only
            return await super().capture_spatial(timings, selection_only=selection_only)

    source = SelectionSource()
    ctx = _context(tmp_path, source)

    async def drive() -> None:
        callback = frame_refresh(ctx.frames, selection_only=True)
        waiting = asyncio.create_task(callback())
        await asyncio.sleep(0)
        request = ctx.frames.pending_spatial_refresh()
        assert request is not None
        await perceive._refresh_spatial_once(ctx, request)
        assert await waiting

    asyncio.run(drive())
    assert source.selection_only is True


def test_fast_hud_observation_publishes_before_full_detection(tmp_path, gates) -> None:
    source = FakeSource()
    ctx = _context(tmp_path, source)

    async def capture_hud(tick, _timings):
        return Sighting(
            Perception(
                tick=tick,
                hud_readings={
                    "food": 200,
                    "wood": 200,
                    "gold": 100,
                    "stone": 200,
                    "population": "4/5",
                    "villagers": 3,
                    "idle_present": True,
                    "idle_count": 3,
                },
                spatial_valid=False,
                hud_only=True,
            )
        )

    source.capture_hud = capture_hud
    _run(perceive.perceive_hud_once(ctx, 1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.hud_only
    assert frame.world is not None and frame.world.villagers == 3
    assert frame.world.spatial_valid is False


def _read_hud(tmp_path) -> LoopContext:
    """One pass over a frame carrying a HUD reading."""
    ctx = _context(
        tmp_path,
        FakeSource([Perception(hud_readings={"food": 120, "wood": 250, "population": "12/20"})]),
    )
    _run(perceive.perceive_once(ctx, tick=1))
    return ctx


def test_a_hud_reading_reaches_the_game_state(tmp_path, gates) -> None:
    assert _read_hud(tmp_path).memory.game_state.resources.get("wood") == 250


def test_capture_crossing_input_cannot_replace_observed_hud(tmp_path, gates) -> None:
    ctx = _context(
        tmp_path,
        FakeSource([Perception(hud_readings={"food": 999}, spatial_valid=False)]),
    )
    before = ctx.memory.game_state.resources["food"]
    _run(perceive.perceive_once(ctx, tick=1))
    assert ctx.memory.game_state.resources["food"] == before
    assert ctx.frames.latest() is None


def test_a_hud_reading_reaches_the_build_gates(tmp_path, gates) -> None:
    """The perceive loop is the only writer of the gates' HUD snapshot."""
    ctx = _read_hud(tmp_path)
    assert ex._build_gates.resources == ctx.memory.game_state.resources


def test_no_entities_means_no_alarm(tmp_path, gates) -> None:
    """An empty frame must not cost an alarm check — it cannot find a threat."""
    ctx = _context(tmp_path, FakeSource([Perception()]))
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.alarm is False


def test_the_alarm_rides_on_the_frame(tmp_path, gates) -> None:
    """Act reads the alarm off the frame, so it must be stamped there, not on
    a variable only the old single tick could see."""
    # 3 is the alarm floor: one stray spearman once rang the town bell and
    # garrisoned the whole economy (exp_0013, turn 14).
    threats = tuple(_ent("knight_line", (100.0, 100.0), f"knight_{i}") for i in range(3))
    owner = _enemy_owner()
    ctx = _context(
        tmp_path,
        FakeSource(
            [Perception(entities=threats)],
            ownership={f"knight_{i}": (owner, 0.9) for i in range(3)},
        ),
    )
    _run(perceive.perceive_once(ctx, tick=1))
    frame = ctx.frames.latest()
    assert frame is not None and frame.alarm is True


def test_the_loop_stops_when_the_game_is_gone(tmp_path, gates, monkeypatch) -> None:
    """Perceive is the loop that can see the window close."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: False)
    ctx = _context(tmp_path, FakeSource())
    _run(perceive.perceive_loop(ctx))
    assert ctx.memory.game_end_reason == "game_not_found"


def _unfocusable(tmp_path, monkeypatch) -> tuple[LoopContext, FakeSource]:
    """A window that never focuses, with the retry wait removed."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: False)
    monkeypatch.setattr(perceive, "_MAX_FOCUS_FAILURES", 3)
    monkeypatch.setattr(perceive, "_FOCUS_RETRY_DELAY", 0.0)
    source = FakeSource()
    ctx = _context(tmp_path, source)
    _run(perceive.perceive_loop(ctx))
    return ctx, source


def test_permanent_focus_loss_ends_the_run_labelled(tmp_path, gates, monkeypatch) -> None:
    """F-1: 12 of 30 iterations went to an unfocusable window, unlabelled."""
    ctx, _source = _unfocusable(tmp_path, monkeypatch)
    assert ctx.memory.game_end_reason == "lost_focus"


def test_an_unplayable_window_is_never_billed_a_frame(tmp_path, gates, monkeypatch) -> None:
    _ctx, source = _unfocusable(tmp_path, monkeypatch)
    assert source.captures == 0


def test_the_loop_resumes_when_focus_returns(tmp_path, gates, monkeypatch) -> None:
    """A transient focus loss must not end the run — it costs frames, not the game."""
    monkeypatch.setattr(perceive, "is_game_running", lambda: True)
    monkeypatch.setattr(perceive.config, "perceive_interval", 0.0)
    focus = iter([False, False, True])
    monkeypatch.setattr(perceive, "ensure_game_focused", lambda: next(focus, True))
    monkeypatch.setattr(perceive, "_FOCUS_RETRY_DELAY", 0.0)
    source = FakeSource()
    ctx = _context(tmp_path, source)

    async def drive() -> None:
        task = asyncio.create_task(perceive.perceive_loop(ctx))
        while source.captures < 2:
            await asyncio.sleep(0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(task, timeout=1.0)

    _run(drive())
    assert source.captures >= 2  # the loop resumed once focus came back


def _enemy_owner():
    """The classifier's enemy label — an ownership map keys the alarm on it."""
    from detection.inference.ownership import Owner

    return Owner.ENEMY
