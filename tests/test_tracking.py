"""Backend choice and the native adapter, checked against the Python reference."""

from __future__ import annotations

import numpy as np
import pytest
from gameplay_agent import tracking
from gameplay_agent.camera_motion import CameraMotion
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.temporal_entities import TemporalEntities, prepare_detections

STABLE = CameraMotion("stable")


def _entity(label: str, box: tuple[float, float, float, float]) -> EntitySnapshot:
    x0, y0, x1, y1 = box
    return EntitySnapshot("raw", label, box, ((x0 + x1) / 2, (y0 + y1) / 2), 0.8)


def _pixels(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (360, 640, 3), dtype=np.uint8)


FRAMES = (
    (_entity("villager", (100, 100, 120, 140)), _entity("house", (300, 100, 360, 160))),
    (_entity("villager", (104, 100, 124, 140)), _entity("house", (300, 100, 360, 160))),
    (_entity("house", (300, 100, 360, 160)),),
    (_entity("villager", (110, 102, 130, 142)), _entity("house", (300, 100, 360, 160))),
)


def _run(tracking_choice: tracking.Tracking) -> list[list[tuple[str, str]]]:
    pixels = _pixels()
    outputs: list[list[tuple[str, str]]] = []
    for capture_id, entities in enumerate(FRAMES, start=1):
        output = tracking_choice.tracker.observe(
            tracking_choice.prepare(entities, pixels),
            capture_id=capture_id,
            captured_at=1.0 + capture_id * 0.2,
            camera_revision=0,
            motion=STABLE,
        )
        outputs.append([(item.id, item.class_name) for item in output])
    return outputs


def test_native_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tracking.BACKEND_ENV, raising=False)
    chosen = tracking.make_tracking()
    assert chosen.backend == "native"
    assert isinstance(chosen.tracker, tracking.NativeTemporalEntities)


@pytest.mark.parametrize(
    ("requested", "backend", "kind"),
    [
        ("python", "python", TemporalEntities),
        ("shadow", "shadow", tracking.ShadowTemporalEntities),
        ("native", "native", tracking.NativeTemporalEntities),
        ("bogus", "native", tracking.NativeTemporalEntities),
    ],
)
def test_the_environment_picks_the_backend(
    monkeypatch: pytest.MonkeyPatch, requested: str, backend: str, kind: type
) -> None:
    monkeypatch.setenv(tracking.BACKEND_ENV, requested)
    chosen = tracking.make_tracking()
    assert chosen.backend == backend
    assert isinstance(chosen.tracker, kind)


def test_a_missing_library_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "_native_available", lambda: False)
    chosen = tracking.make_tracking("native")
    assert chosen.backend == "python"
    assert isinstance(chosen.tracker, TemporalEntities)


def test_native_signatures_match_the_reference() -> None:
    pixels = _pixels(2)
    entities = (*FRAMES[0], _entity("villager", (-20, 10, 15, 40)))
    native = tracking.native_prepare(entities, pixels)
    reference = prepare_detections(entities, pixels)
    assert [item.entity for item in native] == [item.entity for item in reference]
    for ours, theirs in zip(native, reference, strict=True):
        assert ours.appearance == pytest.approx(theirs.appearance, abs=1e-9)


def test_native_tracking_decides_like_the_reference() -> None:
    assert _run(tracking.make_tracking("native")) == _run(tracking.make_tracking("python"))


def test_shadow_acts_on_python_and_counts_agreement() -> None:
    chosen = tracking.make_tracking("shadow")
    assert _run(chosen) == _run(tracking.make_tracking("python"))
    shadow = chosen.tracker
    assert isinstance(shadow, tracking.ShadowTemporalEntities)
    assert (shadow.frames, shadow.differing_frames) == (len(FRAMES), 0)


def test_shadow_counts_a_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = tracking.make_tracking("shadow")
    shadow = chosen.tracker
    assert isinstance(shadow, tracking.ShadowTemporalEntities)
    monkeypatch.setattr(shadow._native, "observe", lambda *_args, **_kwargs: ())  # pyright: ignore[reportPrivateUsage]
    _run(chosen)
    assert shadow.differing_frames == len(FRAMES)
