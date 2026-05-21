"""
Class ID mapping utilities for AoE2 detection class scheme.

The canonical class scheme is defined in classes.yaml (60 classes).
This module provides utilities for loading class definitions,
converting label files, and generating CVAT-compatible class lists.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from .._classes_schema import ClassesYaml

# Root paths
_CONFIG_DIR = Path(__file__).parent.parent / "training" / "config"
_DATASET_YAML = Path(__file__).parent.parent / "training_data" / "dataset.yaml"


def load_classes_yaml(path: Path | None = None) -> dict[int, str]:
    """Load the 55-class target schema from classes.yaml.

    Returns:
        Dict mapping class ID -> class name.
    """
    path = path or (_CONFIG_DIR / "classes.yaml")
    with path.open() as f:
        data = cast("ClassesYaml", yaml.safe_load(f))

    return {entry["id"]: entry["name"] for entry in data["classes"]}


def load_dataset_yaml(path: Path | None = None) -> dict[int, str]:
    """Load the v1 model's class schema from dataset.yaml.

    Returns:
        Dict mapping class ID -> class name.
    """
    path = path or _DATASET_YAML
    with path.open() as f:
        data = cast("dict[str, dict[object, str]]", yaml.safe_load(f))

    return {int(cast("int", k)): v for k, v in data["names"].items()}


def build_v1_to_v2_mapping(
    v1_classes: dict[int, str] | None = None,
    v2_classes: dict[int, str] | None = None,
) -> dict[int, int]:
    """Build mapping from model class IDs to classes.yaml IDs.

    Maps by matching class names between the model's dataset.yaml
    and the canonical classes.yaml scheme. For v5+ models that use
    classes.yaml IDs directly, this returns an identity mapping.

    Returns:
        Dict mapping model_class_id -> classes_yaml_id.
    """
    if v1_classes is None:
        v1_classes = load_dataset_yaml()
    if v2_classes is None:
        v2_classes = load_classes_yaml()

    v2_name_to_id = {name: cid for cid, name in v2_classes.items()}

    mapping = {}
    for v1_id, v1_name in v1_classes.items():
        if v1_name in v2_name_to_id:
            mapping[v1_id] = v2_name_to_id[v1_name]

    return mapping


def get_classes_for_cvat(schema: str = "v2") -> list[str]:
    """Get ordered class names list for CVAT import.

    Args:
        schema: "v1" for 46-class (model), "v2" for 55-class (target).

    Returns:
        List of class names ordered by ID.
    """
    if schema == "v1":
        classes = load_dataset_yaml()
    else:
        classes = load_classes_yaml()

    max_id = max(classes.keys())
    return [classes.get(i, f"unknown_{i}") for i in range(max_id + 1)]


def write_classes_txt(output_path: Path, schema: str = "v2") -> None:
    """Write classes.txt file for CVAT import.

    Args:
        output_path: Where to write the file.
        schema: "v1" or "v2".
    """
    names = get_classes_for_cvat(schema)
    output_path.write_text("\n".join(names) + "\n")


def convert_label_file(
    input_path: Path,
    output_path: Path,
    mapping: dict[int, int],
    skip_unmapped: bool = True,
) -> int:
    """Convert a YOLO label file from one class scheme to another.

    Args:
        input_path: Source label file.
        output_path: Destination label file.
        mapping: Class ID mapping dict.
        skip_unmapped: If True, skip lines with unmapped class IDs.

    Returns:
        Number of labels written.
    """
    lines_out = []
    with input_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            old_id = int(parts[0])
            if old_id in mapping:
                parts[0] = str(mapping[old_id])
                lines_out.append(" ".join(parts))
            elif not skip_unmapped:
                lines_out.append(line.strip())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines_out) + "\n" if lines_out else "")
    return len(lines_out)


if __name__ == "__main__":
    # Print class scheme summary
    classes = load_classes_yaml()
    print(f"Classes (classes.yaml): {len(classes)}")
    for cid, name in sorted(classes.items()):
        print(f"  {cid:3d}: {name}")
