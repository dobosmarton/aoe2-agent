"""Single-source-of-truth for `from ultralytics import YOLO`.

`ultralytics.__all__` does not export `YOLO` directly even though
`YOLO` *is* the documented public entry point (it lives in
`ultralytics.models.yolo.model.YOLO` and is re-exported via
`ultralytics/__init__.py`). Pyright flags every direct
`from ultralytics import YOLO` as `reportPrivateImportUsage`.

Re-exporting once here keeps the suppression on a single line instead
of repeating it across every CLI/training/labeling script.
"""

from ultralytics import YOLO as YOLO  # pyright: ignore[reportPrivateImportUsage]

__all__ = ["YOLO"]
