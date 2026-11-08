"""Read the class names Ultralytics embeds in an exported model.

Ultralytics writes ``names`` into ONNX custom metadata and CoreML user-defined
metadata as the ``repr`` of an ``{id: name}`` dict. Those names belong to the
trained checkpoint, so they outrank a separately maintained ``classes.yaml``:
a 65-class model served with a 60-name list would report ``unknown_60``.
"""

from __future__ import annotations

import ast
from typing import cast


def parse_ultralytics_names(raw: str | None) -> tuple[str, ...] | None:
    """Return names ordered by class ID, or None if `raw` is absent or malformed.

    IDs must be exactly ``0..n-1``; a gap would silently shift every later class.
    """
    if not raw:
        return None
    try:
        parsed = cast("object", ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    entries = cast("dict[object, object]", parsed)
    names: dict[int, str] = {}
    for class_id, name in entries.items():
        if not isinstance(class_id, int) or not isinstance(name, str):
            return None
        names[class_id] = name
    if sorted(names) != list(range(len(names))):
        return None
    return tuple(names[i] for i in range(len(names)))
