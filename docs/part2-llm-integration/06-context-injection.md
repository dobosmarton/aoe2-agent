# Chapter 6: Decision Context

The real-game agent builds routine policy requests from immutable observations. HUD readings retain their capture time, source, and validity; an unreadable field remains unknown rather than becoming a guessed resource balance. Spatial detections carry their own capture and camera revision. The actor revalidates advice against current facts before input.

## Routine decisions

The controller builds a bounded list of named actions from the shared catalog. TypeSafe receives that list with observed state, pending operations, recent failures, goals, and allocation focus. A model response cannot edit observed resources, population, age, or worker counts. Invalid or late advice is replaced by a deterministic choice from the same currently feasible catalog.

A new frame alone does not invalidate advice. Intervening input, an incompatible goal or age, an alarm, or a failed preflight does. The executor checks current selection and target evidence again immediately before committing a command.

## Deliberate context

The deliberate executor is used for combat, tactical handoffs, and bounded recovery. `build_llm_context()` in `apps/agent/src/turn_phases.py` combines working memory, goals, a detected-entity summary, and known/pending building facts. `ExecutorProvider._build_content()` adds capture dimensions and sends this text over the configured `ChatWire`. It does not query the separate `packages/data` SQLite database or invent fallback resource values from text parsing.

Cross-game notes may be included in the cached system prompt by `MemoryChain`. Those notes can influence reasoning but are not authoritative observations. The per-game `ActionLedger` owns purchases and their later reconciliation from screen evidence.

See [Chapter 2](../part1-architecture/02-game-loop-pipeline.md) for scheduling and [Chapter 3](../part1-architecture/03-action-model-and-execution.md) for input preflight.
