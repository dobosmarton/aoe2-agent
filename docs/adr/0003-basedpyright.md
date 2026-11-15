# ADR 0003: basedpyright as the Python type checker

Status: accepted.

## Context

The Python code crosses many SDK boundaries (OpenAI, Anthropic, numpy, redis-py, DuckDB) whose stubs return `Any` in places. Pyright in its default configuration lets `Any` flow on from those calls without a diagnostic, so an untyped value can reach typed code unnoticed.

## Decision

basedpyright is the type checker for the uv workspace. It is a dev dependency (`basedpyright>=1.39.0`) and reads `[tool.pyright]` in the root `pyproject.toml`:

| Setting | Value |
|---|---|
| `include` | `apps/*/src`, `packages/*/src` (tests are not checked) |
| `typeCheckingMode` | `basic` |
| `pythonVersion` | `3.11` |
| `reportAny`, `reportExplicitAny` | `error` |
| `reportMissingImports` | `warning` |
| `reportMissingTypeStubs`, `reportMissingModuleSource` | off |

Any value whose static type is `Any`, inferred or written, is an error. Code at an SDK boundary narrows with `cast(...)` or an annotated local. A per-line `# pyright: ignore[rule]` is the escape hatch where a stub is wrong (44 such comments in first-party source). Hand-written stubs for `coremltools` live in `typings/coremltools/`. `.basedpyright/baseline.json` is checked in and records no files, so every reported error fails the check.

The same command runs in three places: `just typecheck`, the `basedpyright` pre-commit hook and the CI step, all as `uv run basedpyright`.

## Consequences

- Untyped SDK results are visible at the call site as a `cast` or an annotation a reviewer can see.
- Boundary code is more verbose: values that are plainly `int` still need a `cast` when the stub says `Any`.
- `basic` mode leaves the other strict-mode rules off; the Any rules are the only additions on top of it.
- Tests are outside `include`, so they can use untyped helpers freely.

## Alternatives considered

- Pyright: silently ignores `reportAny` and `reportExplicitAny`, which are basedpyright rules and the reason for the choice.
- mypy: its strict mode clashes with much of the pyright-style inference the code is written against.
- `typeCheckingMode = "strict"`: not used. The configuration takes `basic` and adds the Any rules.
