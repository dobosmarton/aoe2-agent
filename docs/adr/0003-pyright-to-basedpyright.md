# ADR 0003 — Switch from pyright to basedpyright with `reportAny` enforcement

**Status:** Accepted (2026-05). Shipped over commits `4f884f0` / `ec4880c` / `7d33a29`.

## Decision

Use [basedpyright](https://github.com/DetachHead/basedpyright) instead of vanilla pyright. Enable `reportAny` repo-wide. Stub third-party libraries that don't ship types (currently `coremltools`) under `typings/`. Use `# pyright: ignore[reportX]` per-line only at SDK boundaries where the upstream stubs are wrong.

## What we considered

1. **Stay on pyright.** No change.
2. **Move to mypy.** Different ecosystem; mypy's strict mode is incompatible with much of pyright's inference style.
3. **basedpyright with `reportAny`.** Stricter superset of pyright; better defaults for catching the long tail of `Any` leaks that vanilla pyright tolerates.

## Why basedpyright

The vanilla pyright config tolerates `Any` flowing freely through SDK boundaries (anthropic, numpy, redis-py, duckdb). That's manageable in a small codebase; in this one it had grown into a baseline of ~2200 violations that nobody checked because they weren't surfaced as errors. Every new piece of code reaching for an SDK was rolling a die on whether its return type was actually typed.

basedpyright's `reportAny` flips this: any expression whose static type is `Any` is an error unless explicitly suppressed. That gives three things:

- **Boundary discipline.** SDK return values get explicit `cast()` or local-variable annotations at the call site — which makes the unsafety visible to reviewers.
- **Local stubs become first-class.** `typings/coremltools/` is a few hand-written stubs that close a real type hole rather than a global `--ignore-missing-imports` switch.
- **Baseline-driven cleanup.** `.basedpyright/baseline.json` lets us ratchet — existing violations are recorded but new ones fail CI.

## What changed in code

- `pyproject.toml` — `[tool.basedpyright]` config, `reportAny: error`, repo-wide scope.
- `.basedpyright/baseline.json` — checked-in baseline. The post-cleanup state is near-zero; new code must add zero new violations.
- `typings/coremltools/` — local stubs replacing the prior `# pyright: ignore[reportMissingImports]` comments. The justification is in [commit `8323bdb`](../../) — the import was guarded behind a try/except for optional installs, but the guard didn't suppress `reportAny`-style errors on the import line itself, so a real stub is cleaner than a per-line ignore.
- `justfile typecheck:` → `basedpyright`.
- CI — `.github/workflows/*` switched to basedpyright; venv path aligned to `venv/` (see commit `77f5a4c`).

## Consequences

**Positive**
- New code is far more likely to have correct types at SDK boundaries.
- The strict-mode-on-by-default story means a contributor doesn't have to know about a separate "strict mode" — there is only one mode.
- Catches subtle bugs at type-check time (numpy stubs returning `Any` from `.shape[i]` is the canonical one — see `apps/arena/src/ranking.py:104`'s carefully-typed BT solver).

**Negative**
- More ceremony at SDK boundaries (explicit `cast("int", ...)` on values that obviously *are* `int`). Cost is paid once per boundary; the value is auditability.
- `basedpyright` is a smaller community than pyright; fewer Stack Overflow answers. Mitigation: most issues are also pyright issues since basedpyright is a fork.
- The baseline file is a checked-in artefact that needs care during rebases. We've found this manageable; the alternative (no baseline, full cleanup in one PR) is worse.

## Related

- [Chapter 1 — System Overview](../part1-architecture/01-system-overview.md) for the broader strictness story.
- Tooling skill: `python-foundations` covers the patterns used across the codebase.
