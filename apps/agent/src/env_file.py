"""The `.env` loader that runs before `config` builds its global.

`config.py` builds `config = Config.from_env()` at import, so every entry point
that imports `config` has already missed its chance to load a file. Loading it
here, from `config` itself, is the one place guaranteed to run first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_HERE = Path(__file__).resolve().parent


def load_env_file(start: Path = _HERE) -> None:
    """Apply the nearest `.env` at or above `start`, or do nothing if none exists.

    An exported variable wins: `setdefault` never overwrites one, so a one-off
    `set AOE2_MODEL=...` still beats the file.
    """
    env_path = _find_env_file(start)
    if env_path is None:
        return
    for key, value in _assignments(env_path.read_text(encoding="utf-8")):
        os.environ.setdefault(key, value)


def _find_env_file(start: Path) -> Path | None:
    """The first `.env` in the `start` directory or an ancestor."""
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _assignments(text: str) -> Iterator[tuple[str, str]]:
    """Every `KEY=value` line. Blank, `#` and no-`=` lines are skipped."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        yield key.strip(), value.strip().strip('"').strip("'")


__all__ = ["load_env_file"]
