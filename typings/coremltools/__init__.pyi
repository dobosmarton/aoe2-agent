"""Minimal type stubs for `coremltools` — just enough surface for
`server/app.py:_load_model`'s CoreML-native loading branch.

coremltools is a macOS-only optional dep (declared in pyproject's
`[coreml]` extra), so it's deliberately not installed on Linux CI.
Without these stubs, every CI typecheck would hit
`reportMissingImports` on the guarded `import coremltools as ct` even
though the runtime import is fenced by `except ImportError`.

We only stub what `server/app.py` actually uses; expand if/when other
callers reference more of the API. Adding a name here is cheap, and
the alternative — sprinkling `# pyright: ignore[reportMissingImports]`
across every callsite — would be noisier and would mask real typos
in module attribute names.

Why stubs over a per-file pragma: stubs let pyright type the
expressions downstream of the import (e.g. `ct.models.MLModel(...)`
returns `MLModel`, not `Any`), which keeps `reportAny` enforcement
intact. A blanket pragma would silence the missing-import warning
but leave every `ct.X.Y` as `Any` and require additional ignores.
"""

from coremltools import models as models

class ComputeUnit:
    """`coremltools.ComputeUnit` — the compute-target enum.

    Only `CPU_AND_NE` is referenced from `server/app.py`. The full enum
    also includes `ALL`, `CPU_ONLY`, `CPU_AND_GPU`; add them here if a
    future caller needs them.
    """

    CPU_AND_NE: ComputeUnit
