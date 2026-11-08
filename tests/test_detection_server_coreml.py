"""The Mac server must ask the CoreML provider for the MLProgram format.

The default NeuralNetwork format stores integer tensors as floats, so the
YOLO26 end-to-end head gathers the wrong top-k anchors. The scores used the
CPU provider, so the served v9 fell to F1 0.282 (CPU 0.842) unnoticed.
"""

from __future__ import annotations

from types import SimpleNamespace

import onnxruntime as ort
from detection_server import app as server


class _FakeSession:
    providers: object = None

    def __init__(self, path: str, options: object, providers: object) -> None:
        _FakeSession.providers = providers

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=[1, 3, 736, 1280])]

    def get_modelmeta(self) -> SimpleNamespace:
        return SimpleNamespace(custom_metadata_map={})


def test_coreml_provider_uses_the_mlprogram_format(monkeypatch) -> None:
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)

    state = server._load_onnx("model.onnx", ("tree",))  # pyright: ignore[reportPrivateUsage]

    assert state.backend == "onnx_coreml"
    assert state.input_hw == (736, 1280)
    assert _FakeSession.providers == [
        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
        "CPUExecutionProvider",
    ]


def test_cpu_only_host_keeps_the_cpu_provider(monkeypatch) -> None:
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)

    state = server._load_onnx("model.onnx", ("tree",))  # pyright: ignore[reportPrivateUsage]

    assert state.backend == "onnx_cpu"
    assert _FakeSession.providers == ["CPUExecutionProvider"]
