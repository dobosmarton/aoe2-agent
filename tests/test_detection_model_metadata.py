"""Class names come from the exported model, not a separately maintained list.

A 65-class checkpoint served with the 60-name fallback list would report
``unknown_60`` for goats, cows and the other appended classes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from detection.inference.model_metadata import parse_ultralytics_names
from detection.training import export_onnx
from detection_server import app as server

_MODELS = Path(__file__).resolve().parents[1] / "packages/detection/src/inference/models"


def test_parses_ultralytics_names_in_id_order() -> None:
    raw = "{1: 'gold_mine', 0: 'tree', 2: 'stone_mine'}"

    assert parse_ultralytics_names(raw) == ("tree", "gold_mine", "stone_mine")


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not a literal",
        "['tree', 'gold_mine']",
        "{}",
        "{0: 'tree', 2: 'stone_mine'}",
        "{'0': 'tree'}",
        "{0: 7}",
    ],
)
def test_rejects_absent_or_malformed_names(raw: str | None) -> None:
    assert parse_ultralytics_names(raw) is None


def test_server_prefers_names_embedded_in_the_model() -> None:
    fallback = ("tree", "gold_mine")
    embedded = "{0: 'tree', 1: 'gold_mine', 2: 'goat'}"

    assert server._model_class_names(embedded, fallback) == ("tree", "gold_mine", "goat")  # pyright: ignore[reportPrivateUsage]


def test_server_falls_back_to_bundled_names() -> None:
    fallback = ("tree", "gold_mine")

    assert server._model_class_names(None, fallback) == fallback  # pyright: ignore[reportPrivateUsage]


def test_reads_names_from_a_bundled_onnx_export() -> None:
    model = _MODELS / "aoe2_yolo_v5.onnx"
    if not model.exists():
        pytest.skip("bundled ONNX weights are not checked out")
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = parse_ultralytics_names(session.get_modelmeta().custom_metadata_map.get("names"))

    assert names is not None
    assert names[0] == "tree"


def _record_exports(monkeypatch) -> list[tuple[str, int | tuple[int, int], bool, bool]]:
    calls: list[tuple[str, int | tuple[int, int], bool, bool]] = []

    def record(model_path: str, imgsz: int | tuple[int, int], dynamic: bool, end2end: bool) -> str:
        calls.append((model_path, imgsz, dynamic, end2end))
        return "unused.onnx"

    monkeypatch.setattr(export_onnx, "export_to_onnx", record)
    return calls


def test_export_defaults_match_the_static_1280_end_to_end_graph(monkeypatch) -> None:
    calls = _record_exports(monkeypatch)
    monkeypatch.setattr("sys.argv", ["export_onnx", "--model", "candidate.pt"])

    export_onnx.main()

    assert calls == [("candidate.pt", 1280, False, True)]


def test_export_takes_a_rectangular_shape_and_the_one_to_many_head(monkeypatch) -> None:
    calls = _record_exports(monkeypatch)

    export_onnx.main(["--model", "candidate.pt", "--imgsz", "736", "1280", "--no-end2end"])

    assert calls == [("candidate.pt", (736, 1280), False, False)]


def test_evaluator_reads_names_from_the_model_file() -> None:
    model = _MODELS / "aoe2_yolo_v5.onnx"
    if not model.exists():
        pytest.skip("bundled ONNX weights are not checked out")
    import onnxruntime as ort
    from detection_server.evaluate import load_cpu_model

    state = load_cpu_model(model)
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get("names")

    assert state.class_names == parse_ultralytics_names(raw)
