# Synthetic Arena: An Analysis of Forkable, Raceable, Mutable Agent Evaluation for AoE2 LLM Arena

**Date**: 2026-05-11
**Author**: Claude (research + analysis)
**Status**: **SUPERSEDED BY IMPLEMENTATION** — fork / race / mutate / observe shipped through Phase 9 plus the broker rollout. Frozen historical analysis; for current state see [Part 6 — Evaluation Arena](../part6-evaluation-arena/14-arena-overview.md) and [Part 7 — Arena Web](../part7-arena-web/19-web-architecture.md).

---

## Context

The AoE2 LLM Arena agent is maturing past the point where single-run, single-config, real-game testing scales. Iterating on prompts, models, perception parameters, or strategist cadence currently requires running a real AoE2 instance in a Windows VM, paying ~$0.50/run for live LLM tests, and reading file-based structlogs after the fact. The autoresearch loop exists and runs sequential games, but it cannot:

1. **Fork**: start N agents from an identical mid-game state and let them diverge.
2. **Race**: run config variants in parallel and pick a winner with statistical rigor.
3. **Mutate**: pause a run, change a world parameter (resources, unit counts, fog), and resume.
4. **Observe / steer**: surface a live web view of agent reasoning, world state, and fork-diffs.

We want all four. The scope of this analysis is a **synthetic perception layer**: the real `gameplay_agent` runtime (detection_phase → strategist_phase → turn_phases → executor) talks to a fake world instead of AoE2.exe. Variants race on **prompts, models, perception, and loop pacing** as a configurable grid.

This document maps the field's best practices to those four capabilities and recommends a concrete architecture rooted in the existing codebase.

---

## Current state — what exists, what's missing

A 60-second tour of what was found in this repo:

| Capability | Already exists | Gap |
|---|---|---|
| Stateful synthetic world | `evaluation/world_sim.py` (resources, ages, villager queue, building costs) | Doesn't emit detections; no perturbation API |
| Synthetic perception | `detection/inference/mock.py` (`mock_detect()` returns frozen Dark Age) | Stateless — no awareness of `world_sim` state |
| Decoupled agent loop | `gameplay_agent/{detection,strategist,turn}_phase.py` (recent refactor) | Phase modules read globals (singleton `config`); no instance scoping |
| Scenario harness | `evaluation/runner.py` + `scenarios/*.yaml` + assertion DSL | One-shot; no fork / branching / pause |
| Experiment orchestrator | `autoresearch/orchestrator.py`, `game_runner.py` | Sequential, single-machine, no ranking |
| Replay log | structlog → `logs/YYYY_MM_DD/game.txt` + optional screenshots | Append-only text; not queryable; not event-sourced |
| Metrics | `AgentMemory.get_metrics_snapshot()` returns 20+ fields | Per-run only, no cross-run aggregation/UI |
| Web UI | Empty `.superset/config.json` placeholder; FastAPI present (for detection server) | No game-state dashboard exists |
| Determinism | `random.seed(42)` in mock_detect | LLM `temperature` is hardcoded SDK default; `random.uniform()` in `executor.py:221` is unseeded |
| Reference paths | `gameplay_agent/main.py`, `gameplay_agent/game_loop.py`, `gameplay_agent/config.py`, `gameplay_agent/memory.py`, `gameplay_agent/providers/claude_tools.py` | — |

The takeaway: this is not a greenfield environment. The shape of the answer is "compose existing parts into a fork-able harness", not "build a simulator from scratch."

---

## Field survey: best practices, with citations

### 1. Forking / branching from a common state

**Canonical patterns**

- **OpenSpiel** ([docs](https://openspiel.readthedocs.io/en/latest/api_reference.html), [serialize ref](https://github.com/google-deepmind/open_spiel/blob/master/docs/api_reference/game_serialize_game_and_state.md)): every `State` has `Clone()` and `serialize_game_and_state()`. This is the gold-standard API surface.
- **Gymnasium / PettingZoo** ([API](https://gymnasium.farama.org/api/env/)): deliberately don't mandate clone — `reset(seed=...)` is the official reproducibility story. Real branching requires custom `get_state()` / `set_state()` per env. Atari's ALE exposes `clone_state()`/`restore_state()` natively.
- **Voyager** ([arxiv 2305.16291](https://arxiv.org/abs/2305.16291)): forks at the *skill* layer — skills are content-addressable code, so swapping a single skill is a free branching primitive. Lesson: separate "agent identity" (memory + skills) from "agent code" so you can mutate code without losing learned state.
- **Generative Agents / Smallville** ([arxiv 2304.03442](https://arxiv.org/abs/2304.03442)): supports URL-addressable timestep replay (`<sim>/<timestep>`) and lets operators "rewrite object state in natural language" mid-replay — exactly the mutation API the user described.
- **OpenAI Five "surgery"** ([Dota 2 paper §4](https://cdn.openai.com/dota-2.pdf)): preserved weights across 20 architecture/rule changes over 10 months instead of retraining (40 months saved). Generalizes to LLM agents: surgery on prompts/tools should not require recreating long-running memory.

**Core invariant**: a forkable simulation must snapshot the tuple `(world_state, agent_state, RNG_state)`. Missing the RNG state turns "deterministic replay" into a lie. For LLM agents the tuple grows: also `(LLM_context, prompt_cache_key)`.

### 2. Multi-variant racing / tournament evaluation

- **AlphaStar League** ([Nature PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)): three roles — main agents (PFSP self-play), main exploiters (find current weaknesses), league exploiters (find global blind spots). Evaluation = round-robin + held-out human-mimic agents. **Prioritized Fictitious Self-Play (PFSP)** weights opponent sampling by historical win-rate against the variant being trained.
- **Population-Based Training (PBT)** ([blog](https://deepmind.google/blog/population-based-training-of-neural-networks/), [arxiv 1711.09846](https://arxiv.org/abs/1711.09846)): train N agents in parallel; periodically the worst N/4 copy weights + hyperparams from the best N/4, then perturb. For LLM agents, "hyperparams" become *prompts, strategist cadence, tool-routing thresholds*.
- **TrueSkill vs Elo**: prefer TrueSkill for evolving stables — it tracks σ (uncertainty) per player, robust to unequal game counts. OpenAI Five used it internally for checkpoint ranking.
- **Bradley-Terry** ([Chatbot Arena](https://www.lmsys.org/blog/2023-12-07-leaderboard/)): LMSYS migrated from Elo for better confidence intervals on pairwise win-rates. The cleanest model for offline ranking of agent variants from observed matches.
- **Inspect AI** ([inspect.aisi.org.uk](https://inspect.aisi.org.uk/), [github](https://github.com/UKGovernmentBEIS/inspect_ai)): now the de facto open LLM-agent eval harness. >200 prebuilt evals, **Inspect View** for trace inspection, sandboxing toolkit, external-agent adapter, static-HTML traces. The pragmatic choice for the orchestration layer.
- **Pitfall — benchmark exploitability**: Berkeley RDI's ["How We Broke Top AI Agent Benchmarks"](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/). Most published numbers are exploitable via prompt-injection or task leakage. Randomize map seeds, civ assignment, starting conditions; never let the agent see oracle state.

### 3. Pause / inject / resume — interactive simulation manipulation

The dominant pattern is **event-sourced replay**: store every action + RNG draw in an append-only log, replay deterministically to any timestep.

- **DeepMind Reverb** ([github](https://github.com/google-deepmind/reverb), [arxiv 2102.04736](https://arxiv.org/abs/2102.04736)) — production-grade C++ + gRPC table-of-trajectories store. Overkill at single-machine scale, but the **abstraction** is right: tuples keyed by `(run_id, timestep)`.
- **Time-travel debugging** (rr, Pernosco) — records syscalls for backward stepping. Relevant only if the agent code itself is buggy; not the right tool for world-state time travel.
- **Chaos engineering for RL agents** ([arxiv 2510.20314](https://arxiv.org/html/2510.20314v1)): minor state perturbations drop DRL agent accuracy by 40–60%. LLM-agent robustness survey: [arxiv 2505.03096](https://arxiv.org/pdf/2505.03096). The "Chaos Orchestrator" pattern (telemetry-driven adversarial action selection) ports cleanly to game-agent stress tests.
- **Game-state perturbation methodology**: paired evaluation — run agent on `(state, state')` differing by a controlled mutation, measure decision divergence.

**Concrete pattern**: event-source every game; periodic full snapshots as cheap insurance against replay drift; `fork(run_id, t, mutation_fn)` API; store in DuckDB/SQLite at single-machine scale.

### 4. Web interfaces for observing / steering

- **Langfuse** ([langfuse.com](https://langfuse.com)) — open-source MIT, OpenTelemetry-native; LLM-as-judge, annotation queues, prompt experiments went open-source mid-2025. **Recommended.** Self-hostable; agent-trace-shaped.
- **LangSmith** — best if LangChain/LangGraph; proprietary.
- **Arize Phoenix** — open-source, strong for RAG; OpenInference standard.
- **Live-debug UI patterns to steal**:
  - **AlphaStar Visualizer** ([starcraft.ai resources](https://starcraft.ai/research/alphastar-resources)): side-by-side game state + NN activations + considered action distribution + predicted outcome.
  - **PySC2 viewer** ([github](https://github.com/google-deepmind/pysc2)): action arrows overlaid directly on the game, camera follows attention.
  - **Smallville web replay**: URL-addressable timestep + agent memory browser; static replay reading event log.
  - **Inspect View**: tool-call tree, per-step tokens, sandbox state — all in a static HTML bundle (no server for post-hoc analysis).
- **Stack rec**: FastAPI + Server-Sent Events + a small React frontend. Three panels: (1) world-state minimap/timeline, (2) LLM trace (prompts, tool calls, raw responses), (3) world diff vs sibling fork. Streamlit works for a v1 but breaks down on bidirectional control; Chainlit is chat-shaped, weak on custom viz.

### 5. Stochasticity & reproducibility — the LLM problem

LLMs are **not deterministic at temperature=0**. Recent results that change how to evaluate:

- [arxiv 2408.04667 "Non-Determinism of Deterministic LLM Settings"](https://arxiv.org/html/2408.04667v5) — even at `temp=0`, 5–12% of prompts give different outputs across seeds.
- [arxiv 2506.09501 "Numerical Sources of Nondeterminism"](https://arxiv.org/html/2506.09501v2) — FP32 near-deterministic, FP16 moderately variable, **BF16 substantially unstable**. GPU floating-point reductions are non-associative.
- [Thinking Machines "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) — root cause is competing logits within numerical noise; batching changes which path wins.
- [arxiv 2601.15322 "Replayable Financial Agents"](https://arxiv.org/html/2601.15322) — **decision determinism** (79–84%) vs **tool-path determinism** (drops to 56.8% on semi-structured tasks). Tool-path variance dominates non-reproducibility in agentic settings.

**Mitigations**, in order of cost:

1. `temperature=0`, fix `seed` where the API supports it (OpenAI + Anthropic both do now).
2. Prompt caching — identical prefix → more consistent outputs + cheaper.
3. Run N trials per condition (N≥20 for ranking; N≥100 to detect <5% deltas). Single-trial benchmarks are unreliable.
4. Log and compare **action sequences**, not only outcomes — win-rate can be stable while paths diverge wildly.
5. Pin model snapshot (`claude-sonnet-4-6`, not floating aliases).

### 6. Prior work directly relevant to AoE2

| System | Domain | Pattern to steal | Source |
|---|---|---|---|
| **PyAge2** | AoE2 (!) | OpenAI-Gym wrapper, DLL injection, 20–30 min game → seconds | [github](https://github.com/kachayev/pyage2) |
| **aoe2-ai-module** | AoE2 DE | Unofficial AI scripting extensions (closest thing to a real API) | [github](https://github.com/FLWL/aoe2-ai-module/) |
| **AlphaStar** | StarCraft II | League + PFSP + exploiter agents + raw-vs-camera replay | [Nature](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) |
| **SIMA / SIMA 2** | 3D-game generalist | 600-skill atomic eval taxonomy, OCR task-completion detection | [SIMA 2 paper](https://arxiv.org/abs/2512.04797) |
| **Voyager** | Minecraft | Skill library as content-addressable code, self-verification loop | [arxiv 2305.16291](https://arxiv.org/abs/2305.16291) |
| **Generative Agents** | Social sandbox | URL-addressable timestep replay, NL world-state injection | [arxiv 2304.03442](https://arxiv.org/abs/2304.03442) |
| **Inspect AI** | LLM-agent harness | Sandboxing, external-agent adapter, static-HTML trace viewer | [inspect.aisi.org.uk](https://inspect.aisi.org.uk/) |
| **Reverb** | RL infra | Table-of-trajectories abstraction, priority sampling | [arxiv 2102.04736](https://arxiv.org/abs/2102.04736) |

PyAge2 is the closest prior art for AoE2 itself — even if not adopted directly, its action-space shaping decisions encode hard-won AoE2-specific lessons.

---

## Recommended architecture for the AoE2 LLM Arena

The scope choice — **synthetic perception layer** — is the right pivot. The agent's whole code path (detection → ownership → context build → LLM call → tool dispatch → executor) runs unchanged; only the bottom layer (screenshot capture + YOLO detection + executor mouse/keyboard sinks) is replaced.

```
┌─────────────────────────────────────────────────────────────┐
│  Arena Controller (new)                                     │
│  - spawns N agent processes per Run                          │
│  - injects WorldState seed + ConfigProfile                  │
│  - subscribes to per-agent event streams                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
   ┌────────▼────┐ ┌──────▼─────┐ ┌──────▼─────┐
   │ Agent A     │ │ Agent B    │ │ Agent C    │
   │ ConfigProf. │ │ ConfigProf.│ │ ConfigProf.│
   │             │ │            │ │            │
   │ game_loop ──┼─┼─ game_loop─┼─┼─ game_loop │ ◀── real code path
   │  ↓ phases   │ │  ↓ phases  │ │  ↓ phases  │
   │ detection_  │ │ detection_ │ │ detection_ │
   │  phase      │ │  phase     │ │  phase     │
   └──┬──────────┘ └──┬─────────┘ └──┬─────────┘
      │ swap          │ swap         │ swap
      ▼               ▼              ▼
   ┌────────────────────────────────────────────┐
   │ SyntheticWorldServer (new)                 │
   │   - per-agent WorldState instance          │
   │   - tick() drives resources/age/queues     │
   │   - render() → DetectedEntity[]            │
   │   - apply_action() consumes agent actions  │
   │   - mutate() applies operator perturbation │
   │   - snapshot()/restore() for fork & resume │
   └────────────────────────────────────────────┘
                           │
                           ▼ event-sourced
   ┌────────────────────────────────────────────┐
   │ Event Log (SQLite/DuckDB)                  │
   │   (run_id, agent_id, t, kind, payload)     │
   └──────────────────────────┬─────────────────┘
                              ▼
   ┌────────────────────────────────────────────┐
   │ FastAPI + SSE + React (new)                │
   │   - live trace per agent                   │
   │   - world-state timeline + minimap         │
   │   - fork diff view                         │
   │   - operator mutation form                 │
   └────────────────────────────────────────────┘
```

### Key design decisions

#### A. Synthetic world = `world_sim.py` + a perception projection

Promote `evaluation/world_sim.py` to a first-class **`SyntheticWorld`** with:

- Existing fields (resources, ages, villager queue, building costs) — keep.
- A `render() → list[DetectedEntity]` method that **projects** world state to detections matching the schema in `detection/inference/detector.py`. This is the missing link between the stateful world and the agent's perception layer.
- `snapshot() / restore()` returning a serialized dict (Pydantic — already idiomatic in the codebase).
- `apply_action(action: Action) → ActionResult` consuming the existing tool schema from `gameplay_agent/providers/claude_tools.py`. The world updates resources/positions accordingly. This replaces `gameplay_agent/executor.py`'s pyautogui sinks.
- `mutate(patch: dict)` for operator perturbations — set resources, hide entities, spawn enemies. Mirrors Smallville's NL injection but typed.

Why this works: `world_sim.py` already models the right state shape for AoE2 Dark→Imperial regression testing. The current `mock_detect()` is a *constant function* of (screenshot dimensions); upgrading it to be a function of `WorldState` is small, additive, and unblocks everything else.

Calibrate `WorldState` constants (unit costs, build times, age requirements, gather rates) from [openage](https://github.com/SFTtech/openage/)'s converted `nyan` data files rather than hand-encoding. openage itself is pre-alpha and not viable as a simulation backend today — *"gameplay is basically non-functional"* per their README — but their asset-converter output of the original AoE2 DAT files is authoritative ground truth for game constants, and depending on it as *data* (not as a runtime) avoids the GPLv3 / C++/Qt/Cython build-complexity contagion that adopting the engine itself would bring.

#### B. Agent process isolation via configuration scoping

The singleton `config` in `gameplay_agent/config.py` is the single biggest blocker to parallel racing. Three options, ranked:

1. **Process-per-agent + env-var injection** (recommended). Each variant is a subprocess with its own env. Already supported by `Config.from_env()`. Lowest code change. Natural fault isolation.
2. **Thread-per-agent + ContextVar scoping**. Faster startup but needs surgical removal of every `from .config import config` import-time read.
3. **Asyncio-per-agent**. Same problem as #2 but with cooperative scheduling.

Pick #1 unless racing 50+ variants on one machine, in which case revisit #2. Pair with **a `ConfigProfile` schema** — a YAML file enumerating model, temperature (newly exposed knob), strategist_interval, detection_imgsz, loop_delay, etc. The autoresearch orchestrator already has YAML scaffolding to extend.

#### C. Event log = OpenTelemetry traces → DuckDB → Langfuse

Replace ad-hoc structlog text with an event-sourced log. Schema:

```
events(run_id, agent_id, t, kind, payload_json, ts)
  kind ∈ { 'turn_start', 'observation', 'llm_prompt', 'llm_response',
           'action', 'action_result', 'world_mutation', 'fork', 'metric' }
```

Backend: DuckDB or SQLite — query-able, no server. Mirror to **Langfuse** (self-hosted, MIT, OTel-native) for LLM-trace observability. This is the table-of-trajectories pattern from Reverb at a single-machine scale.

`forks` and `mutations` are first-class event types — a fork is `(parent_run_id, parent_t) → child_run_id`; the world is reconstructed by replaying events up to `parent_t`, applying any mutation, then continuing.

#### D. Determinism protocol

Exposing controls already implicit in the architecture:

- Expose `temperature` and `seed` in `Config`. Anthropic SDK supports both. Currently hardcoded in `gameplay_agent/providers/claude.py:430-436`.
- Seed `random.uniform()` in `gameplay_agent/executor.py:221-222` (building-placement retry). Today this silently makes runs unrepeatable.
- Pin model snapshot (`claude-sonnet-4-6-2026-XX-XX`, not floating).
- Wire prompt caching (Claude SDK supports it). Same prefix = more stable outputs and cheaper.
- Per condition, N≥20 trials, report mean + 95% CI. Log full action sequence; rank on **decision-path divergence** alongside outcome.

Accept that determinism is asymptotic — `temp=0` flips 5–12% of decisions per [arxiv 2408.04667](https://arxiv.org/html/2408.04667v5). Plan for statistics, not exact replay.

#### E. Ranking and racing — start simple, grow toward AlphaStar

- **v1**: simple grid sweep. Run M conditions × N trials, aggregate `AgentMemory.get_metrics_snapshot()` outputs, plot. `autoresearch/orchestrator.py` is the host.
- **v2**: **Bradley-Terry pairwise ranking** over scenarios where outcomes are head-to-head (or per-scenario composite scores). Pulls in the Chatbot Arena pattern.
- **v3**: **frozen pool** of ~10 historical variants + 3–5 scripted baselines (rush/boom/defense) for opponent diversity. PFSP-style sampling. Inspect AI as the harness for run orchestration, retries, parallelism.

Don't build v3 before v1 is providing signal.

#### F. Web UI — replay-log-driven, three panels

- **Backend**: FastAPI app reading the event log via SSE. The existing `server/app.py` (detection server, FastAPI) is a precedent.
- **Frontend**: small React (or Solid) app, three panels:
  1. **World state**: minimap rendered from `WorldState`, timeline scrubber over `t`.
  2. **Trace**: ordered list of `llm_prompt` / `llm_response` / `action` events; same agent-trace shape Langfuse uses.
  3. **Diff view**: when sibling forks exist, side-by-side world + trace diff.
- **Operator panel** (later): form to apply `mutate()` calls and spawn forks. Schema is the typed perturbation API from §A.

Single biggest leverage point in the UI: **Smallville's pattern of "static replay viewer reading from event log"** — the UI doesn't drive runs, it watches them. Buying that decoupling early avoids socket/lifecycle complexity.

#### G. Chaos mode for robustness testing

A `mutate()` library — destroy random units, fog map regions, swap civs, simulate API latency spikes, inject malformed detections. Run a baseline variant against a chaos schedule and rank by **graceful-degradation metric** (composite score under perturbation / composite score baseline). Plugs directly into the event log as `world_mutation` events.

#### H. Infrastructure & reproducibility

The existing project deliberately runs with minimal infrastructure (pip + justfile + GitHub Actions; native execution split across a Windows VM and a macOS host). The synthetic arena introduces stateful third-party services (LLM-trace store, object storage for replay artifacts) that benefit from containerization without disturbing the existing real-game tier.

**Architectural split**:
- **Real-game tier (existing)**: Windows VM running AoE2.exe + agent, macOS detection server. No Docker. Unchanged.
- **Synthetic-arena tier (new)**: pure Python; no game required; runs anywhere. **Stateful services in Docker, application code native.**

**Containerized services** (`docker-compose.yml`, all images digest-pinned):

| Service | Image | Purpose | Volume |
|---|---|---|---|
| `langfuse-web` + `langfuse-worker` | `langfuse/langfuse:3@sha256:…` | LLM trace UI + OTel ingestion | (uses langfuse-db) |
| `langfuse-db` | `postgres:17@sha256:…` | Langfuse backing store | `langfuse-pg-data` |
| `clickhouse` | `clickhouse/clickhouse-server:24@sha256:…` | Langfuse analytics (required by v3+) | `clickhouse-data` |
| `minio` | `minio/minio:RELEASE.YYYY-MM-DD@sha256:…` | S3-compatible store for replays, screenshots, event-log snapshots | `minio-data` |
| `otel-collector` | `otel/opentelemetry-collector-contrib:0.112@sha256:…` | OpenTelemetry ingestion → Langfuse | (stateless) |

Single bridge network `arena-net`; only Langfuse UI exposed to host by default. Postgres, ClickHouse, MinIO unreachable from host except via the service network.

**Native (uncontainerized) components**:
- Arena controller (Python; subprocess pool of agent processes)
- Agent processes themselves (synthetic-perception mode)
- FastAPI web backend (reads event log; serves SSE)
- React frontend (vite dev server)
- **Event-log file** — DuckDB single file on host filesystem

**Why DuckDB-as-file, not containerized Postgres, for the event log**: the event log is OLAP-shaped (aggregate over millions of `(run_id, t, kind)` rows), single-writer, query-only for the UI. DuckDB outperforms Postgres for this access pattern, is a single file (trivial to back up, commit as fixture, ship as replay artifact), and avoids inter-process I/O. Postgres is already in the stack for Langfuse; reusing it would couple arena tail-latency to Langfuse container health for no benefit.

**Python dependency reproducibility**:
- Adopt **`uv`** (drop-in to existing `pyproject.toml`; faster than pip; modern lockfile semantics).
- Commit `uv.lock`. CI runs `uv lock --locked` to verify it's up-to-date.
- Keep `pyproject.toml` ranges (`anthropic>=0.84.0`, etc.) for human readability; lock pins exact.
- Generate `requirements.txt` from `uv export` for pip-only consumers (the Windows VM may stay on pip).

**Image pinning**:
- Every `image:` directive uses `name:tag@sha256:<digest>`.
- **Renovate** (`.github/renovate.json`) auto-PRs digest bumps weekly.
- CI guard: grep fails if any `image:` line lacks `@sha256:`.

**Secrets & config**:
- `.env.example` documents every required variable (`ANTHROPIC_API_KEY`, `LANGFUSE_SECRET`, `MINIO_ROOT_PASSWORD`, etc.).
- Real `.env` gitignored, injected via compose `${VAR}` interpolation.
- No Vault/sops for v1 — single-developer scope. Revisit if multi-host eval becomes a need.

**Bring-up commands** (justfile additions):

```
just arena-infra-up       # docker compose up -d (services only)
just arena-infra-down     # docker compose down (preserves volumes)
just arena-infra-nuke     # docker compose down -v (DATA LOSS; clean slate)
just arena-infra-logs     # tail logs from all services
just arena-infra-status   # docker compose ps + health-check summary
just arena-up             # arena-infra-up + native arena controller + web UI
```

**CI integration**:
- Existing `.github/workflows/ci.yml` (lint + typecheck + unit test) unchanged — keep PR feedback fast.
- New workflow `arena-integration.yml` runs `docker compose -f docker-compose.ci.yml up` with `tmpfs` volumes, executes a 50-turn synthetic smoke test, asserts events land in DuckDB + traces land in Langfuse. **Opt-in via PR label or nightly schedule** so it doesn't slow every PR.

**Backup & data lifecycle**:
- DuckDB event log: nightly `cp` snapshot → MinIO `event-log-snapshots/YYYY-MM-DD/`.
- Langfuse data: out of scope for v1 — recreatable from event log if needed.
- Replay artifacts: MinIO `replays/{run_id}/` with 30-day lifecycle policy unless promoted to `replays/keep/`.

**Onboarding flow (new contributor)**:

```
git clone …
uv sync                    # installs Python deps from uv.lock
cp .env.example .env       # fill in ANTHROPIC_API_KEY
just arena-infra-up        # docker compose up, ~30s
just arena-smoke           # 50-turn synthetic run, exits when assertions pass
```

Five commands. No manual Postgres / ClickHouse / MinIO install.

---

## Suggested phased build sequence

Ordered for early signal, no premature infrastructure.

| Phase | Capability | Concrete deliverable | Effort |
|---|---|---|---|
| 0 | Infra & reproducibility baseline (see §H) | Migrate to `uv` + commit `uv.lock`; `docker-compose.yml` with digest-pinned Langfuse + Postgres + ClickHouse + MinIO; `.env.example`; Renovate config; six new `just arena-*` targets | 1–2 days |
| 1 | Synthetic world projects to perception | `SyntheticWorld.render() → DetectedEntity[]` in `evaluation/world_sim.py`; `mock_detect()` consults it | 2–3 days |
| 2 | Agent runs against synthetic world | Wire game_loop to use SyntheticWorld in test mode; `executor.py` actions consumed by `world.apply_action()` | 3–5 days |
| 3 | Determinism knobs | Expose `temperature`, `seed` in `Config`; pin model snapshot; seed `random.uniform()` | 1 day |
| 4 | Event log | DuckDB schema + a thin structlog → events writer; replace text logs in test mode | 2–3 days |
| 5 | Fork primitive | `fork(run_id, t, mutation_fn=None) → new run_id`; replay events to `t`, apply mutation, branch | 3–5 days |
| 6 | Multi-process racing | Subprocess pool driven by ConfigProfile YAML; aggregate metrics; simple plots | 3–5 days |
| 7 | Web UI v1 | FastAPI + SSE + minimal React; three panels reading event log | 5–8 days |
| 8 | Bradley-Terry ranking | Pairwise outcome model; per-condition CIs | 2–3 days |
| 9 | Pause / resume / inject UI | Operator panel triggers `mutate()` and forks | 3–5 days |
| 10+ | League / Inspect AI / chaos schedule / Langfuse mirror | As needed | — |
| Watch | Track [openage](https://github.com/SFTtech/openage/) maturity | Revisit as a potential `SyntheticWorld` backend if their simulation reaches alpha and exposes an out-of-process agent API. Until then, consume their `nyan` data files only (see §A). | Ongoing |

Phase 0 is a one-time infra/reproducibility setup that pays back from phase 4 onward. Phases 1–3 unblock everything else and are cheap. Phases 4–5 are the core of the proposal. Phases 6–9 are the user-visible features. Treat phase 10+ as opportunistic.

---

## Future: deployment and competitive multi-agent research

The phased plan above is scoped to **local development** of the synthetic arena. The longer-term direction the local environment is meant to enable:

**Goal**: improve agent configs, system prompts, and settings **without running the actual game**, by racing competitive agent populations in deterministic synthetic environments and ranking the winners.

**Design intent**:

1. **Controlled rounds, varied across rounds.** Within a research round, every competing agent runs against the *same* `SyntheticWorld` seed and trajectory; variation lives only in the `ConfigProfile` (prompt template, model, temperature, strategist cadence, etc.). Across rounds, environments differ — so a config that wins reflects config quality, not lucky environment match. This is a 2-axis experimental design (configs × environments) and the natural progression of the league/PFSP pattern from §E. It also enables statistical decomposition of "due to the config" vs "due to the environment" vs "due to interaction" (e.g., hierarchical Bradley-Terry or two-way ANOVA over the round results).

2. **Continuity with `autoresearch/`.** The existing `autoresearch/orchestrator.py` already runs sequential games and collects metrics; the future arena replaces its execution backend (synthetic instead of real-game), its concurrency model (parallel instead of sequential), and its evaluation (ranked tournament instead of standalone). The conceptual layer — "run experiments, mutate configs, learn what works" — stays the same. Treat the synthetic arena as the next backend for autoresearch, not as a replacement.

3. **Hosted, not laptop-bound.** Phase 0's local docker-compose stack is the *substrate*. The eventual target is a cloud or dedicated-server deployment where rounds run unattended at much higher trial counts. **Same compose file, different `.env`** (cloud-managed Postgres, S3-backed event log, etc.) — the architectural shape is identical; only where the services run changes. This is why phase 0 invests in digest-pinning and `uv.lock` now: lifting the local stack to a remote host should cost ~zero infra surprises.

**Explicitly out of scope until the local arena is built**: cloud deployment, secrets management beyond `.env`, multi-host orchestration, automated prompt mutation, ranked-round scheduling. Designing the cloud or autoresearch-integrated version before the local substrate exists is premature — every design decision there depends on what shape the local primitives end up taking.

**When this section converts from future plan to next iteration**: once the local arena is providing usable ranking signal (after phases 6–8 land). At that point the cost/benefit of remote rounds becomes concrete, and this section gets promoted into a fresh phased plan of its own.

---

## Risks and tradeoffs

1. **Sim-to-real gap.** A synthetic perception layer is intentionally lower fidelity than AoE2.exe. Risk: prompt/strategy variants that win in the synth lose in reality. **Mitigation**: keep the existing real-game test path; run final candidates against AoE2 before declaring victory. The two-tier eval pattern (fast synth + expensive real) is the AlphaStar/OpenAI Five default for a reason.

2. **Determinism is asymptotic.** Even with all the knobs, expect 5–12% per-decision variance and ~20–40% tool-path variance ([arxiv 2601.15322](https://arxiv.org/html/2601.15322)). Don't promise exact replay; promise *statistical* replay over N trials. Build CIs into ranking from day one.

3. **Singleton config refactor.** Phase 6 requires removing `from .config import config` import-time reads scattered across `gameplay_agent/`. Easy to underestimate — grep first. Process-per-agent (env var) sidesteps the deepest refactor; thread/asyncio doesn't.

4. **YOLO detection vs synthetic projection drift.** `SyntheticWorld.render()` must match the real detector's output schema closely or the agent will perceive different worlds in synth vs real. **Mitigation**: write a contract test that runs real detection on a screenshot, then a synthetic render of a near-equivalent world, and asserts the schemas align. The existing `tests/test_detector.py` is the natural home.

5. **Event-log schema lock-in.** Once the UI and ranking depend on the schema, changing it is painful. **Mitigation**: version events from day one (`schema_version` column); use Pydantic for payloads so migration is mechanical.

6. **Web UI scope creep.** "Edit world parameters from the browser" is a feature surface that can grow indefinitely. **Mitigation**: ship a read-only replay viewer first (phase 7). Mutation UI (phase 9) only after the read-only view is in use.

7. **Benchmark exploitability.** Per Berkeley RDI, agents will exploit any shortcut the eval permits ([trustworthy benchmarks](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)). Randomize map seeds, civ assignment, starting resources; never let the agent read state the real game wouldn't expose.

8. **Container image drift.** Digest-pinned images mean upstream security fixes don't land automatically. **Mitigation**: Renovate scheduled weekly with auto-merge for patch/minor digest bumps; manual review for major-version bumps. The opt-in integration CI re-runs on every dependency PR, so a bad bump fails loudly before merge.

---

## Critical files to touch (forward reference for execution)

| Capability | File | Action |
|---|---|---|
| World projection | `evaluation/world_sim.py` | Add `render() → DetectedEntity[]`, `snapshot()`, `restore()`, `mutate()` |
| Synthetic detection | `detection/inference/mock.py` | Accept optional `world: SyntheticWorld`; project state to detections |
| Agent action sink | `gameplay_agent/executor.py` | Pluggable backend: `pyautogui` (real) vs `SyntheticWorld.apply_action()` (test); seed `random.uniform()` |
| Determinism knobs | `gameplay_agent/config.py`, `gameplay_agent/providers/claude.py` | Expose `temperature`, `seed`; pin model snapshot |
| Event log | new `evaluation/event_log.py` (DuckDB) | Schema, writer, replay |
| Fork | new `evaluation/fork.py` | `fork(run_id, t, mutation_fn)` |
| Arena controller | new `arena/controller.py`, `arena/config_profile.py` | Subprocess pool + profile loader |
| Web UI | new `arena/web/` (FastAPI) + `arena/web/ui/` (React) | SSE + three panels |
| Ranking | new `arena/ranking.py` | Bradley-Terry over event log |
| Infra orchestration | new `docker-compose.yml`, `docker-compose.ci.yml` | Digest-pinned services (Langfuse, Postgres, ClickHouse, MinIO, OTel collector) |
| Python lock | `pyproject.toml`, new `uv.lock` | Migrate to `uv`; commit lockfile; CI verifies with `uv lock --locked` |
| Env contract | new `.env.example` | All required vars documented (`ANTHROPIC_API_KEY`, `LANGFUSE_SECRET`, `MINIO_ROOT_PASSWORD`, …) |
| Renovate | new `.github/renovate.json` | Auto-PR for Docker digest + Python lock bumps |
| Integration CI | new `.github/workflows/arena-integration.yml` | Opt-in or nightly; brings up compose stack with `tmpfs` volumes |
| Existing reuse | `evaluation/runner.py`, `evaluation/assertions.py`, `autoresearch/orchestrator.py`, `gameplay_agent/{detection,strategist,turn}_phase.py` | No structural change; consumed unchanged |

---

## Verification (when implementation begins)

End-to-end smoke for the synthetic arena:

0. `just arena-infra-up && just arena-infra-status` — all services healthy within 60s; Langfuse UI reachable at `http://localhost:3000`; MinIO console at `http://localhost:9001`. CI: `uv lock --locked` exits 0; grep guard finds no unpinned digests.
1. `just synth-arena-smoke` — runs two agent variants (same prompt, different temperature) against the same `SyntheticWorld` seed for 100 turns, writes events to DuckDB, prints metric deltas.
2. `just fork-test` — runs agent A to t=50, forks two children with different `loop_delay`, asserts both children produce events with `parent_run_id == A`, asserts child snapshots match A at t=50.
3. `just mutate-test` — runs agent, pauses at t=30, applies `mutate({food: -200})`, resumes, asserts the next `observation` event reflects the mutation.
4. `just web-smoke` — starts FastAPI + UI, drives a 50-turn run, opens browser, asserts SSE stream produces ≥1 event per turn and the timeline panel renders.
5. `pytest evaluation/` — existing scenario regression tests should pass unchanged (the synthetic arena is additive, not a replacement).
6. **Sim-to-real check**: run the same variant against a real AoE2 instance for 1 game; compare turn-1 `DetectedEntity[]` from real detector vs synthetic render of an equivalent Dark Age state; assert schema match.

---

## Implementation runbook

Step-by-step pickup tasks for executing this design. Each entry references the relevant design section(s) and the "Critical files to touch" table; it does not restate design content. Tasks within a phase are typically dependent on the prior task; phases unblock in the order shown in [Suggested phased build sequence](#suggested-phased-build-sequence).

### Phase 0 — Infra & reproducibility baseline

References: §H. Phased table row 0.

**0.1 — Migrate Python deps to `uv` with committed lockfile**
- References: §H ("Python dependency reproducibility")
- Files: `pyproject.toml`, new `uv.lock`, new `requirements.txt` (generated via `uv export`), `justfile`, `.github/workflows/ci.yml`
- Reuse: existing `[dev]`/`[server]`/`[coreml]` extras in `pyproject.toml` — no schema change, only manager change
- Done when: `uv sync` from clean checkout succeeds; `uv lock --locked` step in CI exits 0; `just check` still green; `pip install -r requirements.txt` works as a fallback path for the Windows VM
- Depends on: —

**0.2 — docker-compose for stateful services**
- References: §H ("Containerized services")
- Files: new `docker-compose.yml`, new `docker-compose.ci.yml`, new `.env.example`, `justfile` (six new `arena-*` targets per §H "Bring-up commands")
- Reuse: `server/app.py`'s existing FastAPI deployment pattern as a reference for service health-check shape
- Done when: `just arena-infra-up` brings up Langfuse + Postgres + ClickHouse + MinIO + OTel collector with all images digest-pinned; `just arena-infra-status` reports all healthy within 60s; Langfuse UI reachable at `http://localhost:3000`; MinIO console at `http://localhost:9001`
- Depends on: 0.1

**0.3 — Image-pin enforcement & Renovate**
- References: §H ("Image pinning"), §H ("CI integration"), Risk 8
- Files: new `.github/renovate.json`, new `.github/workflows/arena-integration.yml`, CI guard script (single grep line is sufficient)
- Done when: Renovate opens weekly PRs for Docker digest + `uv.lock` bumps; CI fails if any compose `image:` lacks `@sha256:`; integration smoke runs on nightly schedule or `arena-ci` PR label
- Depends on: 0.2

### Phase 1 — Synthetic world projects to perception

References: §A. Phased table row 1.

**1.1 — `SyntheticWorld.render() → list[DetectedEntity]`**
- References: §A ("Synthetic world = `world_sim.py` + a perception projection"), §D for `snapshot`/`restore`
- Files: `evaluation/world_sim.py` (add `render`, `snapshot`, `restore`, `mutate`), `detection/inference/mock.py` (consult optional `world` argument)
- Reuse: existing `DetectedEntity` schema in `detection/inference/detector.py`; existing `random.seed(42)` pattern in `mock_detect()`
- Done when: `render()` returns a `DetectedEntity[]` whose schema is identical to the real detector's; `tests/test_detector.py` has a contract test asserting schema parity; `snapshot()`/`restore()` round-trip via Pydantic
- Depends on: 0.1

### Phase 2 — Agent runs against synthetic world

References: §A, §B. Phased table row 2.

**2.1 — Pluggable executor sink**
- References: §A (executor sink note), Critical files row "Agent action sink"
- Files: `gameplay_agent/executor.py`
- Done when: `Executor` accepts an injectable backend (`PyAutoGUIBackend` for real, `SyntheticWorldBackend` for test); existing real-game path is byte-for-byte unchanged when no backend is passed; new `tests/test_executor_synth.py` exercises the synth path
- Depends on: 1.1

**2.2 — Wire `game_loop` to the synth path under test mode**
- References: §B ("Agent process isolation via configuration scoping")
- Files: `gameplay_agent/game_loop.py`, `gameplay_agent/{detection,strategist,turn}_phase.py` (no structural change; pass `world` through), `gameplay_agent/config.py` (add `synth_mode: bool`)
- Done when: `AOE2_SYNTH_MODE=1 aoe2-agent --iterations 50` runs end-to-end against a SyntheticWorld with no AoE2.exe, no screenshots, no LLM tool dispatch into pyautogui; agent progresses Dark → Feudal in the synth's state
- Depends on: 2.1

### Phase 3 — Determinism knobs

References: §D. Phased table row 3.

**3.1 — Expose temperature and seed in `Config`**
- References: §D, Critical files row "Determinism knobs"
- Files: `gameplay_agent/config.py` (new fields), `gameplay_agent/providers/claude.py` (lines 430–436: replace hardcoded values), `prompts/`
- Done when: `AOE2_TEMPERATURE=0`, `AOE2_LLM_SEED=42`, and `AOE2_MODEL=claude-sonnet-4-6-2026-XX-XX` all flow through to the Anthropic SDK call; pinned-snapshot model name is the default
- Depends on: —

**3.2 — Seed `random.uniform()` in executor**
- References: §D ("Seed `random.uniform()` in `gameplay_agent/executor.py:221-222`")
- Files: `gameplay_agent/executor.py`
- Done when: building-placement retry is seeded from `Config.placement_seed` or run-id-derived hash; two runs with the same seed produce identical placement attempt sequences
- Depends on: —

### Phase 4 — Event log

References: §C. Phased table row 4.

**4.1 — DuckDB event log schema + structlog adapter**
- References: §C ("Event log = OpenTelemetry traces → DuckDB → Langfuse"), Risk 5
- Files: new `evaluation/event_log.py` (Pydantic event types with `schema_version`; DuckDB writer), structlog config updates in `gameplay_agent/`
- Done when: every turn produces ≥1 event of each kind (`turn_start`, `observation`, `llm_prompt`, `llm_response`, `action`, `action_result`); replay-by-replaying-events reconstructs `WorldState` at any timestep; event schema versioned
- Depends on: 2.2, 3.1

### Phase 5 — Fork primitive

References: §C ("Fork primitive"). Phased table row 5.

**5.1 — `fork(run_id, t, mutation_fn=None) → new_run_id`**
- References: §C, §G (mutation library)
- Files: new `evaluation/fork.py`
- Done when: `fork(A, t=50)` produces a `child_run_id` whose first 50 events are byte-identical to A's; subsequent events diverge only via LLM sampling; `fork(A, t=50, mutation_fn=lambda w: w.set_food(-200))` writes a `world_mutation` event before the child's t=51
- Depends on: 4.1

### Phase 6 — Multi-process racing

References: §B, §E. Phased table row 6.

**6.1 — `ConfigProfile` YAML schema**
- References: §B ("ConfigProfile schema")
- Files: new `arena/config_profile.py`, `arena/profiles/*.yaml`
- Reuse: `autoresearch/`'s existing YAML scaffolding
- Done when: profile loader produces an env-var dict consumable by `Config.from_env()`; tests cover schema validation + env injection
- Depends on: 3.1

**6.2 — Subprocess-pool arena controller**
- References: §B ("Process-per-agent + env-var injection")
- Files: new `arena/controller.py`
- Done when: `just arena-race profiles/v1.yaml` spawns N subprocesses with distinct profiles, streams events to the shared DuckDB, aggregates `AgentMemory.get_metrics_snapshot()` outputs, prints a per-condition mean ± CI table
- Depends on: 6.1, 4.1

### Phase 7 — Web UI v1

References: §F. Phased table row 7.

**7.1 — FastAPI + SSE backend**
- References: §F ("Backend"), Critical files row "Web UI"
- Files: new `arena/web/server.py`
- Reuse: `server/app.py` as a FastAPI deployment-pattern reference
- Done when: SSE endpoint `/events?run_id=X` streams events from DuckDB in order; closing the connection cleanly terminates the read; multiple concurrent clients supported
- Depends on: 4.1

**7.2 — React frontend skeleton (Vite)**
- Files: new `arena/web/ui/` (Vite + React + TypeScript)
- Done when: `just arena-web-dev` launches Vite dev server; basic shell connects to SSE endpoint and renders raw events; production build via `just arena-web-build`
- Depends on: 7.1

**7.3 — Three-panel layout (world / trace / diff)**
- References: §F ("Frontend, three panels")
- Files: `arena/web/ui/src/panels/*`
- Done when: world panel renders minimap from `WorldState`; trace panel orders LLM events; diff panel side-by-sides sibling forks; timeline scrubber works
- Depends on: 7.2

### Phase 8 — Bradley-Terry ranking

References: §E. Phased table row 8.

**8.1 — Bradley-Terry pairwise ranking over event log**
- References: §E ("v2: Bradley-Terry pairwise ranking")
- Files: new `arena/ranking.py`
- Done when: `arena.ranking.rank(profiles=[...], scenarios=[...])` returns per-profile ratings + 95% CIs; output reproducible given same event log
- Depends on: 6.2

### Phase 9 — Pause / resume / inject UI

References: §F (operator panel), §G. Phased table row 9.

**9.1 — Operator mutation panel**
- References: §F, §G
- Files: `arena/web/ui/src/panels/operator.tsx`, `arena/web/server.py` (mutation endpoint)
- Done when: form posts `mutate(patch)` to a running agent; the agent's next observation reflects the mutation; mutation appears in event log as `world_mutation`
- Depends on: 5.1, 7.3

### Phase 10+ — Opportunistic

Per the phased table: League / Inspect AI / chaos schedule / Langfuse mirror. Spawn runbook entries here only when the local arena (phases 0–8) is in active use.

---

## Sources

Forking: [OpenSpiel docs](https://openspiel.readthedocs.io/en/latest/api_reference.html), [OpenSpiel serialize](https://github.com/google-deepmind/open_spiel/blob/master/docs/api_reference/game_serialize_game_and_state.md), [Gymnasium](https://gymnasium.farama.org/api/env/), [PettingZoo](https://github.com/Farama-Foundation/PettingZoo), [RLlib checkpointing](https://docs.ray.io/en/latest/rllib/checkpoints.html).

Tournament/league: [AlphaStar blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/), [AlphaStar Nature PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), [PBT blog](https://deepmind.google/blog/population-based-training-of-neural-networks/), [PBT paper](https://arxiv.org/abs/1711.09846), [Chatbot Arena](https://www.lmsys.org/blog/2023-12-07-leaderboard/), [Inspect AI](https://inspect.aisi.org.uk/), [AgentBench](https://arxiv.org/abs/2308.03688), [Berkeley RDI broken benchmarks](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/).

Pause/inject: [Reverb paper](https://arxiv.org/abs/2102.04736), [Reverb github](https://github.com/google-deepmind/reverb), [DRL adversarial survey](https://arxiv.org/html/2510.20314v1), [LLM-agent robustness](https://arxiv.org/pdf/2505.03096), [Chaos for AI](https://ve3.global/blog/chaos-engineering-for-ai-how-do-we-stress-test-ai-driven-applications).

Stochasticity: [Non-Determinism of Deterministic LLM Settings](https://arxiv.org/html/2408.04667v5), [Numerical Sources of Nondeterminism](https://arxiv.org/html/2506.09501v2), [Defeating Nondeterminism (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [Replayable Financial Agents](https://arxiv.org/html/2601.15322).

Web UIs: [Langfuse](https://langfuse.com), [LLM observability comparison](https://www.getmaxim.ai/articles/top-5-llm-observability-platforms-for-2026/), [AlphaStar visualizer resources](https://starcraft.ai/research/alphastar-resources), [PySC2](https://github.com/google-deepmind/pysc2).

Prior work: [Voyager](https://arxiv.org/abs/2305.16291), [Generative Agents](https://arxiv.org/abs/2304.03442), [OpenAI Five](https://cdn.openai.com/dota-2.pdf), [SIMA 2](https://arxiv.org/abs/2512.04797), [Cicero](https://www.science.org/doi/10.1126/science.ade9097), [TALES](https://arxiv.org/html/2504.14128v3), [PyAge2](https://github.com/kachayev/pyage2), [aoe2-ai-module](https://github.com/FLWL/aoe2-ai-module/).
