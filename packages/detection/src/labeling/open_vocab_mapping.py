"""Build open-vocabulary prompts from classes.yaml and map results back to IDs.

An open-vocab detector takes free-text prompts. We derive prompts per class from
the class name (humanised) plus any civ-specific `examples` (e.g. unique_cavalry
-> Cataphract, Boyar, ...), then map each returned label back to a classes.yaml
ID so the detections drop straight into the existing CVAT/YOLO pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from .._classes_schema import ClassesYaml

_CONFIG_DIR = Path(__file__).parent.parent / "training" / "config"
_DEFAULT_CLASSES_PATH = _CONFIG_DIR / "classes.yaml"


def _humanise(name: str) -> str:
    """Turn a class name like 'scout_line' into a natural phrase 'scout'."""
    return name.removesuffix("_line").replace("_", " ").strip()


def build_class_prompts(path: Path | None = None) -> dict[int, tuple[str, ...]]:
    """Build ordered, de-duplicated text prompts for each classes.yaml class.

    Each class contributes its humanised name plus any `examples`. Civ-specific
    units (the `unique_*` groups) thus get prompts for their concrete names,
    which is exactly where the fixed-vocabulary model struggles.
    """
    path = path or _DEFAULT_CLASSES_PATH
    with path.open() as handle:
        data = cast("ClassesYaml", yaml.safe_load(handle))

    prompts: dict[int, tuple[str, ...]] = {}
    for entry in data["classes"]:
        phrases = [_humanise(entry["name"]), *entry.get("examples", [])]
        ordered_unique = dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip())
        prompts[entry["id"]] = tuple(ordered_unique)
    return prompts


def map_open_vocab_label(label: str, path: Path | None = None) -> int | None:
    """Map an open-vocab detection label to a classes.yaml ID, or None if unknown."""
    return _prompt_index(path).get(label.strip().lower())


def _prompt_index(path: Path | None = None) -> dict[str, int]:
    """Reverse index: lower-cased prompt phrase -> classes.yaml ID."""
    return {
        phrase.lower(): class_id
        for class_id, phrases in build_class_prompts(path).items()
        for phrase in phrases
    }
