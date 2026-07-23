"""The authoritative class catalog, loaded once from `classes.yaml`.

Thin wrapper over `detection.labeling.class_mapping.load_classes_yaml` so the 60
class names have a single source of truth shared with training and inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from detection.labeling.class_mapping import load_classes_yaml

from .domain import ClassId, ClassInfo

if TYPE_CHECKING:
    from pathlib import Path


class ClassCatalog:
    def __init__(self, classes_yaml_path: Path) -> None:
        id_to_name = load_classes_yaml(classes_yaml_path)
        self._classes = tuple(
            ClassInfo(id=class_id, name=name) for class_id, name in sorted(id_to_name.items())
        )
        self._names = dict(id_to_name)
        # Reverse map for the prelabel path (detector emits class names). The last
        # id wins on the rare duplicate name, matching `_names`' forward mapping.
        self._ids = {name: class_id for class_id, name in id_to_name.items()}

    def all(self) -> tuple[ClassInfo, ...]:
        return self._classes

    def name_of(self, class_id: ClassId) -> str:
        return self._names.get(class_id, f"unknown_{class_id}")

    def id_of(self, name: str) -> ClassId | None:
        """Class id for a detector's class name, or None if it isn't in the schema."""
        return self._ids.get(name)

    def has(self, class_id: ClassId) -> bool:
        return class_id in self._names

    def __len__(self) -> int:
        return len(self._classes)
