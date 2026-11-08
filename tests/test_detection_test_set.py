"""The independent test set takes only reviewed frames from unused sessions."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from detection.labeling.build_test_set import build_test_set
from detection_server.evaluate import discover_cases
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def _batch(
    tmp_path: Path,
    *,
    session: str = "unused_session",
    checked: bool = True,
    label: str = "villager",
) -> Path:
    batch = tmp_path / "batch"
    images = batch / "images"
    images.mkdir(parents=True)
    image = images / "frame.jpg"
    Image.new("RGB", (200, 100)).save(image)
    sidecar = {
        "checked": checked,
        "imageWidth": 200,
        "imageHeight": 100,
        "shapes": [
            {
                "label": label,
                "shape_type": "rectangle",
                "points": [[-4, 10], [40, 10], [40, 50], [-4, 50]],
            }
        ],
    }
    (images / "frame.json").write_text(json.dumps(sidecar))
    with (batch / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source_session", "split", "sha256"])
        writer.writeheader()
        writer.writerow(
            {
                "image": "images/frame.jpg",
                "source_session": session,
                "split": "test",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    return batch


def _training_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps({"train_sessions": ["trained"], "validation_sessions": ["selected"]})
    )
    return manifest


def test_reviewed_boxes_become_clipped_yolo_labels(tmp_path: Path) -> None:
    output = tmp_path / "test_data"

    build_test_set(_batch(tmp_path), output, [_training_manifest(tmp_path)])

    label = (output / "test/labels/real_frame.txt").read_text().split()
    assert label[0] == "30"
    assert [float(value) for value in label[1:]] == pytest.approx([0.1, 0.3, 0.2, 0.4])
    assert [case.image.name for case in discover_cases(output, "test")] == ["real_frame.jpg"]


def test_unreviewed_prelabels_are_not_ground_truth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not been marked reviewed"):
        build_test_set(
            _batch(tmp_path, checked=False), tmp_path / "out", [_training_manifest(tmp_path)]
        )


@pytest.mark.parametrize("session", ["trained", "selected"])
def test_sessions_that_trained_or_selected_a_model_are_refused(
    tmp_path: Path, session: str
) -> None:
    with pytest.raises(ValueError, match="already trained or selected"):
        build_test_set(
            _batch(tmp_path, session=session), tmp_path / "out", [_training_manifest(tmp_path)]
        )


def test_fine_labels_become_their_class_and_stay_on_record(tmp_path: Path) -> None:
    output = tmp_path / "test_data"

    build_test_set(_batch(tmp_path, label="ibex"), output, [_training_manifest(tmp_path)])

    assert (output / "test/labels/real_frame.txt").read_text().split()[0] == "5"  # deer
    assert (output / "test/fine_labels/real_frame.txt").read_text() == "ibex\n"
    assert (
        discover_cases(output, "test")[0].fine_labels == output / "test/fine_labels/real_frame.txt"
    )
