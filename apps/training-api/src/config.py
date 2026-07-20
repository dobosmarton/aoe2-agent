"""Process configuration for the training tracker.

All environment reads happen in `load_config` so the rest of the package takes a
plain `TrackerConfig` value — pure, injectable, and trivially testable (no hidden
`os.environ` access buried in call sites).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# The detection package keeps its datasets, raw screenshots, and class schema
# under packages/detection/src. Resolve it relative to this file so the service
# works regardless of the process CWD (uv runs resolve paths from odd places).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DETECTION_SRC = _REPO_ROOT / "packages" / "detection" / "src"

_DEFAULT_DB_PATH = _REPO_ROOT / "logs" / "training" / "tracker.db"
_DEFAULT_DATASET_ROOT = _DETECTION_SRC
_DEFAULT_RAW_IMAGES_DIR = _DETECTION_SRC / "real_screenshots" / "raw"
_DEFAULT_CLASSES_YAML = _DETECTION_SRC / "training" / "config" / "classes.yaml"
_DEFAULT_THUMB_CACHE_DIR = _REPO_ROOT / "logs" / "training" / "thumbs"
_DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://localhost:8100")


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    db_path: Path
    raw_images_dir: Path
    dataset_root: Path
    classes_yaml: Path
    thumb_cache_dir: Path
    cors_origins: tuple[str, ...]


def _path_from(env: Mapping[str, str], key: str, default: Path) -> Path:
    value = env.get(key)
    return Path(value) if value else default


def load_config(env: Mapping[str, str]) -> TrackerConfig:
    """Build a `TrackerConfig` from an environment mapping (usually `os.environ`)."""
    cors_raw = env.get("TRAINING_API_CORS_ORIGINS")
    cors_origins = (
        tuple(origin.strip() for origin in cors_raw.split(",") if origin.strip())
        if cors_raw
        else _DEFAULT_CORS_ORIGINS
    )
    return TrackerConfig(
        db_path=_path_from(env, "TRAINING_API_DB", _DEFAULT_DB_PATH),
        raw_images_dir=_path_from(env, "TRAINING_API_RAW_IMAGES", _DEFAULT_RAW_IMAGES_DIR),
        dataset_root=_path_from(env, "TRAINING_API_DATASET_ROOT", _DEFAULT_DATASET_ROOT),
        classes_yaml=_path_from(env, "TRAINING_API_CLASSES_YAML", _DEFAULT_CLASSES_YAML),
        thumb_cache_dir=_path_from(env, "TRAINING_API_THUMB_CACHE", _DEFAULT_THUMB_CACHE_DIR),
        cors_origins=cors_origins,
    )
