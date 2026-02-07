"""
Class ID mapping between different AoE2 detection class schemes.

The v1 YOLO model was trained on a 46-class scheme (dataset.yaml),
while the target schema is the 55-class scheme (classes.yaml).
This module handles bidirectional mapping between them.
"""

import yaml
from pathlib import Path


# Root paths
_CONFIG_DIR = Path(__file__).parent.parent / "training" / "config"
_DATASET_YAML = Path(__file__).parent.parent / "training_data" / "dataset.yaml"


def load_classes_yaml(path: Path | None = None) -> dict[int, str]:
    """Load the 55-class target schema from classes.yaml.

    Returns:
        Dict mapping class ID -> class name.
    """
    path = path or (_CONFIG_DIR / "classes.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)

    return {entry["id"]: entry["name"] for entry in data["classes"]}


def load_dataset_yaml(path: Path | None = None) -> dict[int, str]:
    """Load the v1 model's class schema from dataset.yaml.

    Returns:
        Dict mapping class ID -> class name.
    """
    path = path or _DATASET_YAML
    with open(path) as f:
        data = yaml.safe_load(f)

    return {int(k): v for k, v in data["names"].items()}


def build_v1_to_v2_mapping(
    v1_classes: dict[int, str] | None = None,
    v2_classes: dict[int, str] | None = None,
) -> dict[int, int]:
    """Build mapping from v1 model class IDs to v2 (classes.yaml) IDs.

    Maps by matching class names between the two schemas.
    Handles special cases where v1 specific unique units map to
    v2 grouped unique classes (e.g., longbowman -> unique_archer).

    Returns:
        Dict mapping v1_class_id -> v2_class_id.
    """
    if v1_classes is None:
        v1_classes = load_dataset_yaml()
    if v2_classes is None:
        v2_classes = load_classes_yaml()

    # Invert v2 to get name -> id
    v2_name_to_id = {name: cid for cid, name in v2_classes.items()}

    # Special mapping for v1 unique units -> v2 grouped unique classes
    _UNIQUE_MAP = {
        "longbowman": "unique_archer",
        "mangudai": "unique_archer",
        "war_wagon": "unique_siege",
    }

    mapping = {}
    for v1_id, v1_name in v1_classes.items():
        if v1_name in v2_name_to_id:
            mapping[v1_id] = v2_name_to_id[v1_name]
        elif v1_name in _UNIQUE_MAP:
            mapped_name = _UNIQUE_MAP[v1_name]
            if mapped_name in v2_name_to_id:
                mapping[v1_id] = v2_name_to_id[mapped_name]

    return mapping


def build_v2_to_v1_mapping(
    v1_classes: dict[int, str] | None = None,
    v2_classes: dict[int, str] | None = None,
) -> dict[int, int]:
    """Build mapping from v2 (classes.yaml) IDs to v1 model IDs.

    Returns:
        Dict mapping v2_class_id -> v1_class_id.
    """
    forward = build_v1_to_v2_mapping(v1_classes, v2_classes)
    return {v2_id: v1_id for v1_id, v2_id in forward.items()}


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
    with open(input_path) as f:
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
    # Print mapping summary
    v1 = load_dataset_yaml()
    v2 = load_classes_yaml()
    mapping = build_v1_to_v2_mapping(v1, v2)

    print(f"V1 classes: {len(v1)}")
    print(f"V2 classes: {len(v2)}")
    print(f"Mapped: {len(mapping)}")
    print()

    print("Mapping (v1 -> v2):")
    for v1_id, v2_id in sorted(mapping.items()):
        print(f"  {v1_id:3d} ({v1[v1_id]:20s}) -> {v2_id:3d} ({v2[v2_id]})")

    # Show v2 classes not in v1
    mapped_v2_ids = set(mapping.values())
    unmapped = {cid: name for cid, name in v2.items() if cid not in mapped_v2_ids}
    if unmapped:
        print(f"\nV2 classes NOT in v1 model ({len(unmapped)}):")
        for cid, name in sorted(unmapped.items()):
            print(f"  {cid:3d}: {name}")
