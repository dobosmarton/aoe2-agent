"""Open-vocabulary detection backends for bootstrapping labels (offline only).

A pluggable backend produces candidate boxes for free-text prompts. `prelabel`
maps the returned labels to classes.yaml IDs and writes CVAT-importable YOLO
labels for human correction — nothing here touches the runtime agent.

Two interchangeable backends sit behind `OpenVocabBackend`:
  - `YoloeBackend`  — local, ONNX-exportable, the default (no hosted API).
  - `DinoXBackend`  — hosted DINO-X Pro (+AP on rare classes); opt-in, API key
                       from the env. Heavier, so guarded behind a separate extra.

Both keep their SDK/network imports lazy so importing this module costs nothing
and needs none of the optional `autolabel` dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class OpenVocabDetection:
    """One open-vocab detection: the matched prompt plus an absolute-pixel box."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    img_size: tuple[int, int]


class OpenVocabBackend(Protocol):
    """A detector that finds boxes for arbitrary text prompts."""

    def detect(self, image_path: Path, prompts: Sequence[str]) -> list[OpenVocabDetection]: ...


class YoloeBackend:
    """Local YOLOE text-prompted detector (the default open-vocab backend)."""

    def __init__(self, model_path: str = "yoloe-11l-seg.pt") -> None:
        self._model_path = model_path

    def detect(self, image_path: Path, prompts: Sequence[str]) -> list[OpenVocabDetection]:
        from detection._ultralytics_compat import YOLO

        names = list(prompts)
        model = YOLO(self._model_path)
        # YOLOE text-prompt flow: register the vocabulary, then predict.
        model.set_classes(names, model.get_text_pe(names))
        results = model(str(image_path), verbose=False)
        return _detections_from_yolo(results[0], names)


class DinoXBackend:
    """Hosted DINO-X Pro detector (opt-in; +AP on rare classes).

    The API key is read from ``DINOX_API_KEY``; the request/response are validated
    with Pydantic at the HTTP boundary and converted to `OpenVocabDetection`.
    """

    _ENDPOINT = "https://api.deepdataspace.com/v2/task/dinox/detection"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("DINOX_API_KEY", "")
        if not self._api_key:
            raise ValueError("DINOX_API_KEY is not set; required for DinoXBackend")

    def detect(self, image_path: Path, prompts: Sequence[str]) -> list[OpenVocabDetection]:
        import base64

        import httpx
        from pydantic import BaseModel

        class _Box(BaseModel):
            category: str
            score: float
            bbox: tuple[float, float, float, float]

        class _Response(BaseModel):
            objects: list[_Box]

        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {"image": image_b64, "prompts": list(prompts)}
        headers = {"Token": self._api_key}
        raw = httpx.post(self._ENDPOINT, json=payload, headers=headers, timeout=120.0)
        raw.raise_for_status()

        parsed = _Response.model_validate(raw.json())
        width, height = _image_size(image_path)
        return [
            OpenVocabDetection(
                label=box.category,
                bbox=box.bbox,
                confidence=box.score,
                img_size=(width, height),
            )
            for box in parsed.objects
        ]


def _detections_from_yolo(result: object, names: Sequence[str]) -> list[OpenVocabDetection]:
    """Convert one ultralytics result into prompt-labelled detections."""
    from detection.inference._ultralytics_results import yolo_boxes_to_lists

    boxes_attr: object | None = getattr(result, "boxes", None)
    if boxes_attr is None or len(cast("list[object]", boxes_attr)) == 0:
        return []

    orig_shape = getattr(result, "orig_shape", (0, 0))
    height = int(cast("int", orig_shape[0]))
    width = int(cast("int", orig_shape[1]))

    bboxes, class_ids, confidences = yolo_boxes_to_lists(boxes_attr)
    return [
        OpenVocabDetection(
            label=names[int(class_id)],
            bbox=(box[0], box[1], box[2], box[3]),
            confidence=confidence,
            img_size=(width, height),
        )
        for box, class_id, confidence in zip(bboxes, class_ids, confidences, strict=True)
        if 0 <= int(class_id) < len(names)
    ]


def _image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return (image.width, image.height)
