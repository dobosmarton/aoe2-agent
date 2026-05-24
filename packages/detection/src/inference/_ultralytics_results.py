"""Typed extractors for ultralytics inference results.

The ultralytics `Boxes` object exposes `.xyxy`, `.cls`, `.conf` as
torch tensors. The chain `.cpu().numpy()` lands in numpy, and then
`numpy.ndarray.__getitem__` / `.tolist()` are typed as `Any` in the
public stubs. Crossing that boundary once here means callers iterate
plain `list[float]` instead of repeating the cast-and-suppress dance
at every detection site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np


def yolo_boxes_to_lists(
    boxes: object,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Pull (xyxy, cls, conf) out of an ultralytics `Boxes` object.

    Returns:
        (bboxes, class_ids, confidences) where:
          - bboxes is `list[list[float]]` of (x1, y1, x2, y2) rows
          - class_ids and confidences are parallel `list[float]`

    Caller is responsible for upstream None-check on `result.boxes`.
    """
    # `boxes` is typed `object` so callers don't need to suppress reportAny
    # when passing in `getattr(result, "boxes", None)`. The attribute access
    # below is the single point where we cross the untyped ultralytics
    # boundary into typed Python primitives.
    xyxy = cast("np.ndarray", boxes.xyxy.cpu().numpy())  # pyright: ignore[reportAttributeAccessIssue]
    cls = cast("np.ndarray", boxes.cls.cpu().numpy())  # pyright: ignore[reportAttributeAccessIssue]
    conf = cast("np.ndarray", boxes.conf.cpu().numpy())  # pyright: ignore[reportAttributeAccessIssue]
    return (
        cast("list[list[float]]", xyxy.tolist()),
        cast("list[float]", cls.tolist()),
        cast("list[float]", conf.tolist()),
    )
