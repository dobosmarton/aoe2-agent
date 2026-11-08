"""train_yolo passes every run-defining setting explicitly.

With ``optimizer=auto`` Ultralytics chose AdamW(lr=0.000145) for v11 and run E,
but MuSGD(lr=0.01) for run D, only because D's dataset crossed 10,000
iterations. A resume must continue the run, not start a new one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
from detection.training import train_yolo


class _FakeYolo:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    def __init__(self, path: str) -> None:
        self.path = path

    def train(self, **kwargs: object) -> None:
        _FakeYolo.calls.append((self.path, kwargs))
        if not kwargs.get("resume"):
            weights = Path(str(kwargs["project"])) / str(kwargs["name"]) / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "last.pt").write_bytes(b"")


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    _FakeYolo.calls = []
    # The test environment has no ultralytics; main() imports YOLO from here.
    compat = ModuleType("detection._ultralytics_compat")
    compat.YOLO = _FakeYolo  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "detection._ultralytics_compat", compat)
    data = tmp_path / "dataset.yaml"
    data.write_text("names: {0: tree}\n")
    return data


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr("sys.argv", ["train_yolo", *argv])
    return train_yolo.main()


def test_new_run_sets_optimizer_seed_and_no_early_stop(
    monkeypatch: pytest.MonkeyPatch, fake_yolo: Path, tmp_path: Path
) -> None:
    code = _run(
        monkeypatch, "--data", str(fake_yolo), "--name", "candidate", "--project", str(tmp_path)
    )

    assert code == 0
    [(model, kwargs)] = _FakeYolo.calls
    assert model == "yolo26n.pt"
    assert kwargs["optimizer"] == "MuSGD"
    assert kwargs["lr0"] == 0.01
    assert kwargs["momentum"] == 0.9
    assert kwargs["warmup_bias_lr"] == 0.0
    assert kwargs["seed"] == 0
    assert kwargs["deterministic"] is True
    assert kwargs["patience"] == 0
    assert kwargs["save_period"] == -1


def test_resume_continues_the_interrupted_run(
    monkeypatch: pytest.MonkeyPatch, fake_yolo: Path, tmp_path: Path
) -> None:
    last = tmp_path / "candidate" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"")

    code = _run(
        monkeypatch,
        "--data",
        str(fake_yolo),
        "--name",
        "candidate",
        "--project",
        str(tmp_path),
        "--resume",
        "--workers",
        "4",
    )

    assert code == 0
    [(model, kwargs)] = _FakeYolo.calls
    assert model == str(last)
    assert kwargs == {"resume": True, "workers": 4, "cache": False}


def test_resume_without_a_checkpoint_fails(
    monkeypatch: pytest.MonkeyPatch, fake_yolo: Path, tmp_path: Path
) -> None:
    code = _run(
        monkeypatch,
        "--data",
        str(fake_yolo),
        "--name",
        "missing",
        "--project",
        str(tmp_path),
        "--resume",
    )

    assert code == 1
    assert _FakeYolo.calls == []
