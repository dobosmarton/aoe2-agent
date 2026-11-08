"""
Class ID mapping utilities for AoE2 detection class scheme.

The canonical class scheme is defined in classes.yaml.
This module provides utilities for loading class definitions
and generating CVAT-compatible class lists.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from .._classes_schema import ClassesYaml

# Root paths
_CONFIG_DIR = Path(__file__).parent.parent / "training" / "config"


def load_classes_yaml(path: Path | None = None) -> dict[int, str]:
    """Load the current class schema from classes.yaml.

    Returns:
        Dict mapping class ID -> class name.
    """
    path = path or (_CONFIG_DIR / "classes.yaml")
    with path.open() as f:
        data = cast("ClassesYaml", yaml.safe_load(f))

    return {entry["id"]: entry["name"] for entry in data["classes"]}


def get_classes_for_cvat() -> list[str]:
    """Get ordered class names list for CVAT import (classes.yaml order).

    Returns:
        List of class names ordered by ID.
    """
    classes = load_classes_yaml()
    max_id = max(classes.keys())
    return [classes.get(i, f"unknown_{i}") for i in range(max_id + 1)]


def load_label_aliases(path: Path | None = None) -> dict[str, str]:
    """Map each fine annotation label to its training class name.

    An alias must be a new name that maps onto an existing class; otherwise a
    reviewer's label could silently shadow or invent a class.
    """
    path = path or (_CONFIG_DIR / "classes.yaml")
    with path.open() as f:
        data = cast("ClassesYaml", yaml.safe_load(f))
    names = {entry["name"] for entry in data["classes"]}
    aliases: dict[str, str] = {}
    for alias in data.get("label_aliases", []):
        label, target = alias["label"], alias["maps_to"]
        if label in names or label in aliases or target not in names:
            raise ValueError(f"Label alias {label!r} -> {target!r} must add a label to a class")
        aliases[label] = target
    return aliases


def get_annotation_labels() -> list[str]:
    """Labels a reviewer may choose: every class in ID order, then each alias."""
    return [*get_classes_for_cvat(), *load_label_aliases()]


def model_classes_match_schema_prefix(model_classes: Sequence[str]) -> bool:
    """A checkpoint may support the unchanged prefix of a newer training schema."""
    classes = load_classes_yaml()
    return bool(model_classes) and all(
        classes.get(index) == name for index, name in enumerate(model_classes)
    )


def write_classes_txt(output_path: Path) -> None:
    """Write classes.txt file for CVAT import."""
    output_path.write_text("\n".join(get_classes_for_cvat()) + "\n")


if __name__ == "__main__":
    # Print class scheme summary
    classes = load_classes_yaml()
    print(f"Classes (classes.yaml): {len(classes)}")
    for cid, name in sorted(classes.items()):
        print(f"  {cid:3d}: {name}")
