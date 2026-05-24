"""Stub for `coremltools.models` — only `MLModel` is referenced.

Kept intentionally minimal:
  * `__init__` signature mirrors how `server/app.py:216` constructs it
    (positional `model_path`, keyword `compute_units`).
  * `predict` / `get_spec` are stubbed because the existing
    `_CoreMLModel` Protocol in `server/app.py:167` lists them — keeping
    the stub consistent with the Protocol surface avoids surprise type
    drift if a future caller switches from the Protocol to the
    concrete `MLModel` type.
"""

class MLModel:
    def __init__(
        self,
        model_path: str,
        compute_units: object = ...,
    ) -> None: ...
    def predict(self, data: dict[str, object]) -> dict[str, object]: ...
    def get_spec(self) -> object: ...
