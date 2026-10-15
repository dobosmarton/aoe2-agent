"""Record a short, native-pixel game sequence for camera/perception replay.

This standalone command does not control the game or block the running agent.
An optional input-event JSONL file uses the same monotonic clock on the VM:
``{"at_monotonic": 123.4, "kind": "camera_jump"}``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .camera_motion import MotionLimits, analysis_pixels, cursor_exclusions, estimate_motion
from .screen import NativeCapture, capture_native

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


@dataclass(frozen=True, slots=True)
class InputEvent:
    at_monotonic: float
    kind: str


def load_input_events(path: Path | None) -> tuple[InputEvent, ...]:
    """Read only explicitly timestamped input events; never infer keypresses."""
    if path is None:
        return ()
    events: list[InputEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = cast("object", json.loads(line))
        if not isinstance(value, dict):
            raise ValueError("input event must be a JSON object")
        raw = cast("dict[str, object]", value)
        at = raw.get("at_monotonic")
        kind = raw.get("kind")
        if not isinstance(at, (int, float)) or not isinstance(kind, str) or not kind:
            raise ValueError("input event needs at_monotonic and kind")
        events.append(InputEvent(float(at), kind))
    return tuple(sorted(events, key=lambda event: event.at_monotonic))


def record_sequence(
    directory: Path,
    *,
    count: int,
    interval: float,
    input_events: tuple[InputEvent, ...] = (),
    grab: Callable[[], NativeCapture] = capture_native,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Save native PNGs and frame/camera/input metadata at start-to-start cadence."""
    if count <= 0 or interval <= 0:
        raise ValueError("count and interval must be positive")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "sequence.jsonl"
    limits = MotionLimits()
    previous: np.ndarray | None = None
    previous_at = float("-inf")
    previous_cursor: tuple[int, int] | None = None
    next_start = clock()
    with manifest.open("w", encoding="utf-8") as output:
        for index in range(count):
            sleep(max(0.0, next_start - clock()))
            capture = grab()
            if capture.window_rect is None:
                raise RuntimeError("AoE2 window not found; sequence capture will not use a monitor")
            analysis = analysis_pixels(capture.rgb, capture.width, capture.height, limits)
            if capture.captured_at - previous_at > 1.0:
                previous = None
            motion = estimate_motion(
                previous,
                analysis,
                full_size=(capture.width, capture.height),
                limits=limits,
                excluded_boxes=(
                    *cursor_exclusions(previous_cursor, limits),
                    *cursor_exclusions(capture.cursor, limits),
                ),
            )
            name = f"frame_{index:05d}.png"
            capture.image().save(directory / name, format="PNG")
            events = [
                event.kind
                for event in input_events
                if previous_at < event.at_monotonic <= capture.captured_at
            ]
            output.write(
                json.dumps(
                    {
                        "capture_id": index,
                        "captured_at_monotonic": capture.captured_at,
                        "width": capture.width,
                        "height": capture.height,
                        "window_rect": capture.window_rect,
                        "cursor": capture.cursor,
                        "image": name,
                        "camera_motion": motion.kind,
                        "camera_shift": motion.shift,
                        "camera_reason": motion.reason,
                        "input_events": events,
                    }
                )
                + "\n"
            )
            output.flush()
            previous = analysis
            previous_at = capture.captured_at
            previous_cursor = capture.cursor
            next_start = max(next_start + interval, clock())
    return manifest


class _Args(argparse.Namespace):
    output: Path
    count: int
    interval: float
    input_events: Path | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--input-events", type=Path)
    args = parser.parse_args(namespace=_Args())
    path = record_sequence(
        args.output,
        count=args.count,
        interval=args.interval,
        input_events=load_input_events(args.input_events),
    )
    print(f"Captured {args.count} frames in {path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
