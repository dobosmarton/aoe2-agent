"""Session provenance keeps development data separate from the sealed test."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from detection.labeling.class_mapping import load_classes_yaml
from detection.labeling.session_dataset import (
    Split,
    label_path,
    list_sessions,
    load_reviewed_labels,
    seal_reviewed_labels,
)
from detection.testing.evaluate_real import Prediction
from detection_server.app import ModelState
from detection_server.evaluate_sessions import evaluate_sessions
from detection_server.session_tools import (
    audit_dataset,
    import_corrected_coco,
    prelabel_session,
    render_qa_gallery,
)
from gameplay_agent.capture_dataset import capture_game_window, capture_session
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def _capture(
    root: Path,
    session_id: str,
    split: Split,
    color: str,
    *,
    append: bool = False,
    stage: str = "dark_age",
    zoom: str = "default",
) -> None:
    capture_session(
        root,
        session_id,
        split,
        "live_game",
        "Highland",
        "Magyars",
        "game-build-1",
        "high",
        "100%",
        stage,
        zoom,
        count=1,
        interval=0,
        start_delay=0,
        append=append,
        grab=lambda: Image.new("RGB", (64, 48), color),
    )


def _coco(directory: Path, names: list[str], *, box: bool = True) -> Path:
    data = {
        "categories": [{"id": 1, "name": "sheep"}],
        "images": [
            {"id": index, "file_name": name, "width": 64, "height": 48}
            for index, name in enumerate(names, start=1)
        ],
        "annotations": (
            [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10]}] if box else []
        ),
    }
    path = directory / "corrected.coco.json"
    path.write_text(json.dumps(data))
    return path


def test_capture_keeps_a_match_together_across_stages_and_zoom(tmp_path: Path) -> None:
    _capture(tmp_path, "match_001", "development", "green")
    _capture(
        tmp_path,
        "match_001",
        "development",
        "blue",
        append=True,
        stage="castle_age",
        zoom="max",
    )

    directory, session = list_sessions(tmp_path)[0]
    assert directory.name == "match_001"
    assert session.split == "development"
    assert [(frame.game_stage, frame.zoom) for frame in session.frames] == [
        ("dark_age", "default"),
        ("castle_age", "max"),
    ]
    assert all((directory / frame.image).suffix == ".png" for frame in session.frames)

    with pytest.raises(ValueError, match="same match"):
        capture_session(
            tmp_path,
            "match_001",
            "final_test",
            "live_game",
            "Highland",
            "Magyars",
            "game-build-1",
            "high",
            "100%",
            "imperial_age",
            "default",
            1,
            0,
            0,
            append=True,
            grab=lambda: Image.new("RGB", (64, 48), "red"),
        )


def test_identical_images_cannot_cross_splits(tmp_path: Path) -> None:
    _capture(tmp_path, "match_dev", "development", "green")
    _capture(tmp_path, "match_test", "final_test", "green")

    with pytest.raises(ValueError, match="Identical image bytes"):
        list_sessions(tmp_path)


def test_final_test_cannot_be_prelabelled_or_scored_early(tmp_path: Path) -> None:
    _capture(tmp_path, "match_test", "final_test", "red")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="annotated blind"):
        prelabel_session(tmp_path, "match_test", model)
    with pytest.raises(ValueError, match="sealed"):
        evaluate_sessions(model, tmp_path, "final_test")


def test_corrected_coco_requires_all_images_and_seals_reviewed_bytes(tmp_path: Path) -> None:
    _capture(tmp_path, "match_dev", "development", "green")
    _capture(tmp_path, "match_dev", "development", "blue", append=True)
    directory, session = list_sessions(tmp_path)[0]

    incomplete = _coco(directory, ["frame_00000.png"])
    with pytest.raises(ValueError, match="every captured image"):
        import_corrected_coco(tmp_path, session.session_id, incomplete)

    complete = _coco(directory, ["task\\frame_00000.png", "frame_00001.png"])
    assert import_corrected_coco(tmp_path, session.session_id, complete) == 1
    assert label_path(directory, session.frames[1]).read_text() == ""
    assert label_path(directory, session.frames[0]).read_text().startswith("8 ")
    assert len(render_qa_gallery(tmp_path, tmp_path / "qa_before_seal")) == 2
    with pytest.raises(FileExistsError, match="Unsealed labels exist"):
        import_corrected_coco(tmp_path, session.session_id, complete)
    assert import_corrected_coco(tmp_path, session.session_id, complete, replace_unsealed=True) == 1
    assert len(list((directory / "annotation_history").glob("labels_*"))) == 1

    reviewed = seal_reviewed_labels(directory, "human-reviewer")
    assert len(reviewed.labels) == 2
    assert load_reviewed_labels(directory, session) == reviewed
    audit = audit_dataset(tmp_path)
    assert audit["labeled_boxes"] == {"development": {"sheep": 1}}
    assert audit["critical_class_support"]["development"]["sheep"] == 1
    assert audit["critical_class_support"]["development"]["farm"] == 0
    assert audit["maps"] == {"development": {"Highland": 1}}
    assert len(render_qa_gallery(tmp_path, tmp_path / "qa_after_seal")) == 2

    with pytest.raises(ValueError, match="Finish all capture batches"):
        _capture(tmp_path, "match_dev", "development", "red", append=True)

    label_path(directory, session.frames[0]).write_text("8 0.1 0.1 0.1 0.1\n")
    with pytest.raises(ValueError, match="bytes changed"):
        load_reviewed_labels(directory, session)


def test_out_of_bounds_label_cannot_be_sealed(tmp_path: Path) -> None:
    _capture(tmp_path, "match_dev", "development", "green")
    directory, session = list_sessions(tmp_path)[0]
    path = label_path(directory, session.frames[0])
    path.parent.mkdir()
    path.write_text("8 0.95 0.5 0.2 0.2\n")

    with pytest.raises(ValueError, match="Box leaves image"):
        seal_reviewed_labels(directory, "human-reviewer")


def test_prelabels_use_current_predictor_without_marking_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture(tmp_path, "match_dev", "development", "green")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    class_names = tuple(load_classes_yaml().values())
    monkeypatch.setattr(
        "detection_server.session_tools.load_cpu_model",
        lambda _: ModelState("onnx_cpu", object(), class_names, str(model)),
    )
    monkeypatch.setattr(
        "detection_server.session_tools.predict",
        lambda *_: [Prediction(8, (-5, 10, 20, 20), 0.9)],
    )

    path = prelabel_session(tmp_path, "match_dev", model)
    prelabels = json.loads(path.read_text())
    assert prelabels["annotations"][0]["category_id"] == 9
    assert prelabels["annotations"][0]["bbox"] == [0.0, 10, 20.0, 10]
    assert prelabels["info"]["model_sha256"]
    assert audit_dataset(tmp_path)["unreviewed_sessions"] == ["match_dev"]


def test_explicit_final_test_uses_the_frozen_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture(tmp_path, "match_test", "final_test", "red")
    directory, session = list_sessions(tmp_path)[0]
    path = label_path(directory, session.frames[0])
    path.parent.mkdir()
    path.write_text("")
    seal_reviewed_labels(directory, "human-reviewer")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    class_names = tuple(load_classes_yaml().values())
    monkeypatch.setattr(
        "detection_server.evaluate_sessions.load_cpu_model",
        lambda _: ModelState("onnx_cpu", object(), class_names, str(model)),
    )
    monkeypatch.setattr("detection_server.evaluate_sessions.collect_samples", lambda *_: [([], [])])

    report = evaluate_sessions(model, tmp_path, "final_test", attest_isolated_final_test=True)

    assert report["evaluation_kind"] == "session_held_out_real_frame_test"
    assert report["session_ids"] == ["match_test"]
    assert report["images"] == 1


def test_capture_initializes_dpi_before_reading_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    initialized = False

    class FakeScreen:
        def __enter__(self) -> FakeScreen:
            nonlocal initialized
            initialized = True
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def grab(self, region: dict[str, int]) -> SimpleNamespace:
            assert region == {"left": 10, "top": 20, "width": 2, "height": 2}
            return SimpleNamespace(size=(2, 2), bgra=bytes((0, 0, 255, 255)) * 4)

    def game_window_rect() -> tuple[int, int, int, int]:
        assert initialized
        return (10, 20, 2, 2)

    monkeypatch.setattr("gameplay_agent.capture_dataset.mss.MSS", FakeScreen)
    monkeypatch.setattr("gameplay_agent.capture_dataset.get_game_window_rect", game_window_rect)

    image = capture_game_window()

    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_capture_requires_the_game_window(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeScreen:
        def __enter__(self) -> FakeScreen:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("gameplay_agent.capture_dataset.mss.MSS", FakeScreen)
    monkeypatch.setattr("gameplay_agent.capture_dataset.get_game_window_rect", lambda: None)

    with pytest.raises(RuntimeError, match="game window not found"):
        capture_game_window()
