"""Capture lossless, session-scoped AoE2 frames for detector evaluation.

Run this as a separate process on the Windows game VM. The exact game window is
required; unlike routine agent capture, this command never falls back to a
monitor screenshot that might include a different UI or scale.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import mss
from detection.labeling.session_dataset import (
    REVIEWED_MANIFEST,
    CaptureFrame,
    CaptureSession,
    Source,
    Split,
    load_session,
    new_session,
    sha256_file,
    utc_now,
    verify_capture,
    write_session,
)
from PIL import Image

from .window import get_game_window_rect

if TYPE_CHECKING:
    from collections.abc import Callable


_ROOT = (
    Path(__file__).resolve().parents[3] / "packages/detection/src/real_screenshots/quality_sessions"
)


def capture_game_window() -> Image.Image:
    # MSS makes the process DPI-aware on Windows. Read the window rectangle only
    # after that happens, so its coordinates match the physical capture pixels.
    with mss.MSS() as screen:
        rect = get_game_window_rect()
        if rect is None:
            raise RuntimeError("AoE2 game window not found; capture will not use the whole monitor")
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid AoE2 window rectangle: {rect}")
        screenshot = screen.grab({"left": left, "top": top, "width": width, "height": height})
    return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def add_frame(
    directory: Path,
    session: CaptureSession,
    image: Image.Image,
    game_stage: str,
    zoom: str,
    captured_at: str,
) -> CaptureSession:
    """Save one native PNG and atomically add its provenance to the manifest."""
    if image.size != (session.width, session.height):
        raise ValueError(f"Window size changed during capture: {image.size}")
    if not game_stage.strip() or not zoom.strip():
        raise ValueError("Game stage and zoom are required for each frame")
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    path = images / f"frame_{len(session.frames):05d}.png"
    if path.exists():
        raise FileExistsError(path)
    image.save(path, format="PNG")
    frame = CaptureFrame(
        image=path.relative_to(directory).as_posix(),
        image_sha256=sha256_file(path),
        captured_at=captured_at,
        game_stage=game_stage,
        zoom=zoom,
    )
    updated = replace(session, frames=(*session.frames, frame))
    write_session(directory, updated)
    return updated


def capture_session(
    root: Path,
    session_id: str,
    split: Split,
    source: Source,
    map_name: str,
    civilization: str,
    game_version: str,
    graphics_preset: str,
    ui_scale: str,
    game_stage: str,
    zoom: str,
    count: int,
    interval: float,
    start_delay: float,
    append: bool = False,
    grab: Callable[[], Image.Image] = capture_game_window,
) -> CaptureSession:
    if count <= 0 or interval < 0 or start_delay < 0:
        raise ValueError("Count must be positive; intervals and delay cannot be negative")
    if start_delay:
        time.sleep(start_delay)
    first = grab()
    first_captured_at = utc_now()
    expected = new_session(
        session_id,
        split,
        source,
        map_name,
        civilization,
        game_version,
        graphics_preset,
        ui_scale,
        first.width,
        first.height,
    )
    directory = root / session_id
    if append:
        session = load_session(directory)
        if replace(expected, frames=session.frames) != session:
            raise ValueError("An appended batch must keep the same match and capture settings")
        if (
            (directory / REVIEWED_MANIFEST).exists()
            or (directory / "prelabels.coco.json").exists()
            or (directory / "labels").exists()
        ):
            raise ValueError("Finish all capture batches before prelabeling or annotation")
        verify_capture(directory, session)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        session = expected
        write_session(directory, session)
    session = add_frame(directory, session, first, game_stage, zoom, first_captured_at)
    for _ in range(1, count):
        time.sleep(interval)
        image = grab()
        session = add_frame(directory, session, image, game_stage, zoom, utc_now())
    return session


class _Args(argparse.Namespace):
    root: Path
    session_id: str
    split: Split
    source: Source
    map_name: str
    civilization: str
    game_version: str
    graphics_preset: str
    ui_scale: str
    game_stage: str
    zoom: str
    count: int
    interval: float
    start_delay: float
    append: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--split", choices=("development", "final_test"), required=True)
    parser.add_argument("--source", choices=("live_game", "replay", "scenario"), required=True)
    parser.add_argument("--map", dest="map_name", required=True)
    parser.add_argument("--civilization", required=True)
    parser.add_argument("--game-version", required=True)
    parser.add_argument("--graphics-preset", required=True)
    parser.add_argument("--ui-scale", required=True)
    parser.add_argument("--game-stage", required=True)
    parser.add_argument("--zoom", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(namespace=_Args())
    session = capture_session(
        args.root,
        args.session_id,
        args.split,
        args.source,
        args.map_name,
        args.civilization,
        args.game_version,
        args.graphics_preset,
        args.ui_scale,
        args.game_stage,
        args.zoom,
        args.count,
        args.interval,
        args.start_delay,
        append=args.append,
    )
    print(f"Captured {len(session.frames)} frames in {args.root / session.session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
