"""Fine reviewer labels map onto existing classes and never shadow one."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from detection.labeling.class_mapping import (
    get_annotation_labels,
    get_classes_for_cvat,
    load_label_aliases,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_schema_aliases_map_onto_existing_classes() -> None:
    aliases = load_label_aliases()

    assert aliases == {"ibex": "deer", "elephant": "boar"}
    assert get_annotation_labels() == [*get_classes_for_cvat(), "ibex", "elephant"]


@pytest.mark.parametrize(
    "alias",
    [
        "{label: deer, maps_to: boar}",  # shadows a class
        "{label: ibex, maps_to: antelope}",  # target is not a class
    ],
)
def test_aliases_must_add_a_label_to_a_real_class(tmp_path: Path, alias: str) -> None:
    schema = tmp_path / "classes.yaml"
    schema.write_text(
        "classes:\n  - {id: 0, name: deer}\n  - {id: 1, name: boar}\n"
        f"label_aliases:\n  - {alias}\n"
    )

    with pytest.raises(ValueError, match="must add a label to a class"):
        load_label_aliases(schema)
