"""Session-scoped gameplay captures and manually reviewed detection labels.

The capture manifest fixes a session's split before annotation. Reviewed labels
have a separate manifest so model-generated prelabels cannot be scored by mistake.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from PIL import Image

from .class_mapping import load_classes_yaml

Split = Literal["development", "final_test"]
Source = Literal["live_game", "replay", "scenario"]
SCHEMA_VERSION = 1
CAPTURE_MANIFEST = "capture.json"
REVIEWED_MANIFEST = "reviewed_annotations.json"
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}\Z")
_CLASS_SCHEMA = Path(__file__).resolve().parents[1] / "training/config/classes.yaml"


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    image: str
    image_sha256: str
    captured_at: str
    game_stage: str
    zoom: str


@dataclass(frozen=True, slots=True)
class CaptureSession:
    schema_version: int
    session_id: str
    split: Split
    source: Source
    map_name: str
    civilization: str
    game_version: str
    graphics_preset: str
    ui_scale: str
    width: int
    height: int
    class_schema_sha256: str
    frames: tuple[CaptureFrame, ...]


@dataclass(frozen=True, slots=True)
class ReviewedLabel:
    image: str
    label: str
    label_sha256: str


@dataclass(frozen=True, slots=True)
class ReviewedAnnotations:
    schema_version: int
    session_id: str
    class_schema_sha256: str
    reviewer: str
    reviewed_at: str
    labels: tuple[ReviewedLabel, ...]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_schema_sha256() -> str:
    return sha256_file(_CLASS_SCHEMA)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object with string keys")
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return cast("list[object]", value)


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty string: {key}")
    return value


def _integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer: {key}")
    return value


def _utc_timestamp(data: dict[str, object], key: str) -> str:
    value = _string(data, key)
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp: {key}") from exc
    offset = stamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"Timestamp must be UTC: {key}")
    return value


def _digest(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256 digest: {value}")
    return value


def _relative_path(directory: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    resolved = (directory / candidate).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"Path escapes session directory: {name}")
    return resolved


def _read_json(path: Path) -> dict[str, object]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(data)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def new_session(
    session_id: str,
    split: Split,
    source: Source,
    map_name: str,
    civilization: str,
    game_version: str,
    graphics_preset: str,
    ui_scale: str,
    width: int,
    height: int,
) -> CaptureSession:
    if _SESSION_ID.fullmatch(session_id) is None:
        raise ValueError("Session ID must use 3-64 letters, digits, '-' or '_'")
    if split not in ("development", "final_test"):
        raise ValueError(f"Unsupported split: {split}")
    if source not in ("live_game", "replay", "scenario"):
        raise ValueError(f"Unsupported source: {source}")
    if split == "final_test" and source == "scenario":
        raise ValueError("Staged scenarios cannot be final-test sessions")
    if (
        width <= 0
        or height <= 0
        or not all((map_name, civilization, game_version, graphics_preset, ui_scale))
    ):
        raise ValueError("Session geometry and game metadata are required")
    return CaptureSession(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        split=split,
        source=source,
        map_name=map_name,
        civilization=civilization,
        game_version=game_version,
        graphics_preset=graphics_preset,
        ui_scale=ui_scale,
        width=width,
        height=height,
        class_schema_sha256=class_schema_sha256(),
        frames=(),
    )


def write_session(directory: Path, session: CaptureSession) -> None:
    if directory.name != session.session_id:
        raise ValueError("Session ID must match its directory name")
    _write_json(directory / CAPTURE_MANIFEST, asdict(session))


def load_session(directory: Path) -> CaptureSession:
    data = _read_json(directory / CAPTURE_MANIFEST)
    frames = tuple(
        CaptureFrame(
            image=_string(frame, "image"),
            image_sha256=_digest(_string(frame, "image_sha256")),
            captured_at=_utc_timestamp(frame, "captured_at"),
            game_stage=_string(frame, "game_stage"),
            zoom=_string(frame, "zoom"),
        )
        for frame in (_mapping(item) for item in _list(data.get("frames")))
    )
    split = _string(data, "split")
    source = _string(data, "source")
    if split not in ("development", "final_test") or source not in (
        "live_game",
        "replay",
        "scenario",
    ):
        raise ValueError("Invalid session split or source")
    session = new_session(
        session_id=_string(data, "session_id"),
        split=cast("Split", split),
        source=cast("Source", source),
        map_name=_string(data, "map_name"),
        civilization=_string(data, "civilization"),
        game_version=_string(data, "game_version"),
        graphics_preset=_string(data, "graphics_preset"),
        ui_scale=_string(data, "ui_scale"),
        width=_integer(data, "width"),
        height=_integer(data, "height"),
    )
    if (
        _integer(data, "schema_version") != SCHEMA_VERSION
        or _digest(_string(data, "class_schema_sha256")) != session.class_schema_sha256
        or directory.name != session.session_id
    ):
        raise ValueError(f"Session schema, class mapping, or path changed: {directory}")
    if not frames:
        raise ValueError(f"Session has no captured frames: {directory}")
    names = [frame.image for frame in frames]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate frame path in session: {directory}")
    return CaptureSession(
        schema_version=session.schema_version,
        session_id=session.session_id,
        split=session.split,
        source=session.source,
        map_name=session.map_name,
        civilization=session.civilization,
        game_version=session.game_version,
        graphics_preset=session.graphics_preset,
        ui_scale=session.ui_scale,
        width=session.width,
        height=session.height,
        class_schema_sha256=session.class_schema_sha256,
        frames=frames,
    )


def verify_capture(directory: Path, session: CaptureSession) -> None:
    for frame in session.frames:
        path = _relative_path(directory, frame.image)
        if path.suffix.lower() != ".png" or sha256_file(path) != frame.image_sha256:
            raise ValueError(f"Capture bytes changed: {path}")
        with Image.open(path) as image:
            if image.size != (session.width, session.height):
                raise ValueError(f"Capture resolution changed: {path}")
            image.verify()


def list_sessions(root: Path) -> list[tuple[Path, CaptureSession]]:
    sessions = [
        (path.parent, load_session(path.parent)) for path in sorted(root.glob("*/capture.json"))
    ]
    if not sessions:
        raise ValueError(f"No session manifests in {root}")
    digests: dict[str, Split] = {}
    for directory, session in sessions:
        verify_capture(directory, session)
        for frame in session.frames:
            previous = digests.setdefault(frame.image_sha256, session.split)
            if previous != session.split:
                raise ValueError(
                    "Identical image bytes appear in development and final-test sessions"
                )
    return sessions


def label_path(directory: Path, frame: CaptureFrame) -> Path:
    return _relative_path(directory, f"labels/{Path(frame.image).stem}.txt")


def validate_yolo_label(path: Path, class_count: int) -> tuple[int, ...]:
    """Validate every box, including boxes that would extend outside the image."""
    classes: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label at {path}:{line_number}")
        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(part) for part in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Malformed label at {path}:{line_number}") from exc
        if not 0 <= class_id < class_count or not all(
            math.isfinite(value) for value in (cx, cy, width, height)
        ):
            raise ValueError(f"Invalid class or coordinate at {path}:{line_number}")
        # COCO-to-YOLO decimal serialization may move an edge by <1e-9.
        tolerance = 1e-8
        if (
            width <= 0
            or height <= 0
            or not (
                -tolerance <= cx - width / 2 < cx + width / 2 <= 1 + tolerance
                and -tolerance <= cy - height / 2 < cy + height / 2 <= 1 + tolerance
            )
        ):
            raise ValueError(f"Box leaves image at {path}:{line_number}")
        classes.append(class_id)
    return tuple(classes)


def seal_reviewed_labels(directory: Path, reviewer: str) -> ReviewedAnnotations:
    """Call only after a human has checked all visible objects, including misses."""
    destination = directory / REVIEWED_MANIFEST
    if destination.exists():
        raise FileExistsError(f"Reviewed labels already sealed: {destination}")
    if not reviewer.strip():
        raise ValueError("A human reviewer name is required")
    session = load_session(directory)
    verify_capture(directory, session)
    class_count = len(load_classes_yaml())
    labels: list[ReviewedLabel] = []
    for frame in session.frames:
        path = label_path(directory, frame)
        if not path.is_file():
            raise ValueError(f"Missing reviewed label file, including empty scenes: {path}")
        validate_yolo_label(path, class_count)
        labels.append(
            ReviewedLabel(frame.image, path.relative_to(directory).as_posix(), sha256_file(path))
        )
    reviewed = ReviewedAnnotations(
        SCHEMA_VERSION,
        session.session_id,
        session.class_schema_sha256,
        reviewer,
        utc_now(),
        tuple(labels),
    )
    _write_json(destination, asdict(reviewed))
    return reviewed


def load_reviewed_labels(directory: Path, session: CaptureSession) -> ReviewedAnnotations:
    data = _read_json(directory / REVIEWED_MANIFEST)
    labels = tuple(
        ReviewedLabel(
            image=_string(label, "image"),
            label=_string(label, "label"),
            label_sha256=_digest(_string(label, "label_sha256")),
        )
        for label in (_mapping(item) for item in _list(data.get("labels")))
    )
    if (
        _integer(data, "schema_version") != SCHEMA_VERSION
        or _string(data, "session_id") != session.session_id
        or _digest(_string(data, "class_schema_sha256")) != session.class_schema_sha256
    ):
        raise ValueError(f"Reviewed label manifest does not match capture session: {directory}")
    if len(labels) != len(session.frames) or {item.image for item in labels} != {
        frame.image for frame in session.frames
    }:
        raise ValueError(f"Reviewed labels do not cover every frame: {directory}")
    class_count = len(load_classes_yaml())
    frames_by_image = {frame.image: frame for frame in session.frames}
    for label in labels:
        path = _relative_path(directory, label.label)
        if path != label_path(directory, frames_by_image[label.image]):
            raise ValueError(f"Reviewed label belongs to a different frame: {label.image}")
        if sha256_file(path) != label.label_sha256:
            raise ValueError(f"Reviewed label bytes changed: {path}")
        validate_yolo_label(path, class_count)
    return ReviewedAnnotations(
        SCHEMA_VERSION,
        session.session_id,
        session.class_schema_sha256,
        _string(data, "reviewer"),
        _utc_timestamp(data, "reviewed_at"),
        labels,
    )
