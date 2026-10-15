"""Short sequence capture preserves timestamps and optional input provenance."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from gameplay_agent.capture_sequence import InputEvent, load_input_events, record_sequence
from gameplay_agent.screen import NativeCapture

if TYPE_CHECKING:
    from pathlib import Path


def test_sequence_has_start_to_start_times_and_camera_events(tmp_path: Path) -> None:
    now = 0.0
    started: list[float] = []

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def grab() -> NativeCapture:
        started.append(now)
        return NativeCapture(bytes(64 * 36 * 3), 64, 36, now, (10, 20, 64, 36))

    path = record_sequence(
        tmp_path / "sequence",
        count=3,
        interval=0.2,
        input_events=(InputEvent(0.1, "camera_jump"), InputEvent(0.35, "selection")),
        grab=grab,
        clock=clock,
        sleep=sleep,
    )

    frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert started == pytest.approx([0.0, 0.2, 0.4])
    assert [frame["input_events"] for frame in frames] == [[], ["camera_jump"], ["selection"]]
    assert [frame["camera_motion"] for frame in frames] == ["reset", "stable", "stable"]
    assert all((path.parent / frame["image"]).exists() for frame in frames)


def test_sequence_rejects_monitor_fallback(tmp_path: Path) -> None:
    def grab() -> NativeCapture:
        return NativeCapture(bytes(64 * 36 * 3), 64, 36, 1.0, None)

    with pytest.raises(RuntimeError, match="window not found"):
        record_sequence(tmp_path / "sequence", count=1, interval=0.2, grab=grab)


def test_input_event_file_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"at_monotonic": 1.2, "kind": "camera_jump"}\n', encoding="utf-8")
    assert load_input_events(path) == (InputEvent(1.2, "camera_jump"),)
    path.write_text('{"at_monotonic": "yesterday", "kind": "camera_jump"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="at_monotonic"):
        load_input_events(path)
