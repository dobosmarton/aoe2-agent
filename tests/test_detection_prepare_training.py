"""Dataset preparation safeguards that affect training labels and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING

import pytest
import yaml
from detection.labeling.prepare_training import prepare_training
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def _source_batch(root: Path, *, empty: bool = False) -> tuple[Path, Path, Path]:
    images = root / "images"
    images.mkdir(parents=True)
    export = root / "export"
    export.mkdir()
    manifest = root / "manifest.csv"
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []
    for index, session in enumerate(("game_a", "game_a", "game_b", "game_c")):
        name = f"frame_{index}.png"
        image = images / name
        Image.new("RGB", (100, 100), (index, index, index)).save(image)
        (images / f"frame_{index}.json").write_text(json.dumps({"checked": True}))
        coco_images.append({"id": index, "file_name": name, "width": 100, "height": 100})
        if not empty or index != 3:
            annotations.append(
                {"id": index, "image_id": index, "category_id": 5, "bbox": [10, 20, 30, 40]}
            )
        rows.append(
            {
                "image": f"images/{name}",
                "source_session": session,
                "split": "development",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "width": "100",
                "height": "100",
            }
        )
    (export / "annotations.json").write_text(
        json.dumps(
            {
                "images": coco_images,
                "annotations": annotations,
                "categories": [{"id": 5, "name": "town_center"}],
            }
        )
    )
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return images, export, manifest


def _prepare(root: Path, images: Path, export: Path, manifest: Path, **kwargs: object):
    return prepare_training(
        [export],
        images_dir=images,
        manifest_paths=[manifest],
        output_dir=root / "dataset",
        **kwargs,
    )


def test_session_split_and_empty_reviewed_frame_are_preserved(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path, empty=True)
    summary = _prepare(tmp_path, images, export, manifest, val_split=0.5, oversample_real=2)
    output = tmp_path / "dataset"
    records = json.loads((output / "dataset_manifest.json").read_text())["images"]
    train_sessions = {item["source_session"] for item in records if item["split"] == "train"}
    val_sessions = {item["source_session"] for item in records if item["split"] == "val"}
    real_train = [item for item in records if item["split"] == "train" and item["source"] == "real"]
    assert not train_sessions & val_sessions
    assert summary.real_train_unique + summary.real_val == 4
    assert len(real_train) == summary.real_train_on_disk
    assert len(list((output / "train/images").iterdir())) == summary.total_train_on_disk
    assert any(not path.read_text() for path in output.glob("*/labels/real_frame_3.txt"))
    assert "path" not in yaml.safe_load((output / "dataset.yaml").read_text())
    assert any(
        path.read_text().startswith("9 ") for path in output.glob("*/labels/real_frame_0.txt")
    )


def test_dry_run_writes_nothing_and_existing_output_is_rejected(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    summary = _prepare(tmp_path, images, export, manifest, dry_run=True)
    assert summary.real_train_unique + summary.real_val == 4
    assert not (tmp_path / "dataset").exists()
    _prepare(tmp_path, images, export, manifest)
    with pytest.raises(FileExistsError, match="already exists"):
        _prepare(tmp_path, images, export, manifest)


def test_invalid_annotation_and_unreviewed_image_fail_before_output(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    path = export / "annotations.json"
    data = json.loads(path.read_text())
    data["annotations"][0]["bbox"] = [90, 20, 30, 40]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid box"):
        _prepare(tmp_path, images, export, manifest)
    assert not (tmp_path / "dataset").exists()
    data["annotations"][0]["bbox"] = [10, 20, 30, 40]
    path.write_text(json.dumps(data))
    (images / "frame_0.json").write_text(json.dumps({"checked": False}))
    with pytest.raises(ValueError, match="not been marked reviewed"):
        _prepare(tmp_path, images, export, manifest)


def test_final_test_session_is_excluded(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    rows = list(csv.DictReader(manifest.open()))
    rows[0]["split"] = "final_test"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Only development frames"):
        _prepare(tmp_path, images, export, manifest)


def test_explicit_synthetic_source_remaps_ids_and_uses_real_validation(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    synthetic = tmp_path / "synthetic"
    (synthetic / "dataset.yaml").parent.mkdir(exist_ok=True)
    (synthetic / "dataset.yaml").write_text(
        yaml.safe_dump({"names": {0: "town_center", 1: "tree"}})
    )
    for split in ("train", "val"):
        image_dir = synthetic / split / "images"
        label_dir = synthetic / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        Image.new("RGB", (100, 100)).save(image_dir / "sample.png")
        (label_dir / "sample.txt").write_text("0 0.500000 0.500000 0.300000 0.400000\n")
    summary = _prepare(tmp_path, images, export, manifest, synthetic_dir=synthetic)
    output = tmp_path / "dataset"
    assert summary.synthetic_train == 1
    assert summary.synthetic_val_diagnostic == 1
    assert summary.validation_images == summary.real_val
    assert (output / "train/labels/synth_sample.txt").read_text().startswith("9 ")
    assert (output / "synthetic_val/labels/synth_sample.txt").is_file()
    assert not (output / "val/labels/synth_sample.txt").exists()


def test_unknown_synthetic_class_is_rejected_before_copying(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    (synthetic / "dataset.yaml").write_text(yaml.safe_dump({"names": {0: "not_a_game_class"}}))
    for split in ("train", "val"):
        image_dir = synthetic / split / "images"
        label_dir = synthetic / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        Image.new("RGB", (100, 100)).save(image_dir / "sample.png")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    with pytest.raises(ValueError, match="absent from training schema"):
        _prepare(tmp_path, images, export, manifest, synthetic_dir=synthetic)
    assert not (tmp_path / "dataset").exists()


def test_historical_real_images_are_deduplicated_and_train_only(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "dataset.yaml").write_text(
        yaml.safe_dump({"names": {0: "tree", 1: "town_center"}})
    )
    for split, color, class_id in (("train", 90, 1), ("val", 180, 0)):
        image_dir = historical / split / "images"
        label_dir = historical / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        Image.new("RGB", (100, 100), (color, 0, 0)).save(image_dir / "real_old.png")
        (label_dir / "real_old.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
    # Historical source split names are not globally unique in this fixture.
    (historical / "val/images/real_old.png").rename(historical / "val/images/real_other.png")
    (historical / "val/labels/real_old.txt").rename(historical / "val/labels/real_other.txt")
    (historical / "train/images/real_old__dup1.png").write_bytes(
        (historical / "train/images/real_old.png").read_bytes()
    )
    (historical / "train/labels/real_old__dup1.txt").write_bytes(
        (historical / "train/labels/real_old.txt").read_bytes()
    )

    summary = _prepare(tmp_path, images, export, manifest, historical_real_dir=historical)
    output = tmp_path / "dataset"
    records = json.loads((output / "dataset_manifest.json").read_text())["images"]
    historical_records = [item for item in records if item["source"] == "historical_real"]
    assert summary.historical_train_unique == 2
    assert summary.real_train_unique == summary.reviewed_train_unique + 2
    assert {item["split"] for item in historical_records} == {"train"}
    assert len(historical_records) == 2
    assert (output / "train/labels/historical_real_old.txt").read_text().startswith("9 ")
    assert (output / "train/labels/historical_real_other.txt").read_text().startswith("0 ")


def test_changed_historical_duplicate_is_rejected(tmp_path: Path) -> None:
    images, export, manifest = _source_batch(tmp_path)
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "dataset.yaml").write_text(yaml.safe_dump({"names": {0: "tree"}}))
    for split in ("train", "val"):
        image_dir = historical / split / "images"
        label_dir = historical / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        Image.new("RGB", (100, 100), (100 if split == "train" else 200, 0, 0)).save(
            image_dir / f"real_{split}.png"
        )
        (label_dir / f"real_{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (historical / "train/images/real_train__dup1.png").write_bytes(
        (historical / "train/images/real_train.png").read_bytes()
    )
    (historical / "train/labels/real_train__dup1.txt").write_text("0 0.6 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="oversample differs"):
        _prepare(tmp_path, images, export, manifest, historical_real_dir=historical)
    assert not (tmp_path / "dataset").exists()
