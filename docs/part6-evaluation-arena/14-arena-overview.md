# Chapter 14 — Arena Overview

The `apps/arena/src/` package is the **synthetic evaluation tier**: it runs the agent against an in-memory AoE2-lite world (`packages/evaluation/src/world_sim.py`) instead of the real game. It exists so we can iterate on prompts, models, and decision logic without booting a Windows VM or paying live-game prices, and so we can rank variants against each other with statistical rigour.

<aside class="prereqs">

[Chapter 1 — System Overview](../part1-architecture/01-system-overview.md) for the real-game tier this synthetic tier is testing. CLI fluency (click-style commands) for the `python -m arena` interface.

</aside>

## When you'd use it

| You want to … | Reach for | Cost | API key? |
|---|---|---|---|
| Sanity-check a code change end-to-end with no spend | `python -m arena smoke` | $0 | No |
| Race two prompt/model variants on the real API | `python -m arena race [profile.yaml]` | ~$0.02–$0.20 | Yes |
| Pick a winner between N variants with 95% CIs | `python -m arena rank [profile.yaml]` | ~$1–$5 (default config: ~$1.20) | Yes |
| Replay any past run visually | `python -m arena_web` + browser | $0 | No |

All three commands live behind one CLI entry point in `apps/arena/src/__main__.py:181`, and all three persist their full event log into a single per-run DuckDB file under `logs/arena/<YYYY-MM-DD>/<label>-<HHMMSS>.duckdb` (`apps/arena/src/__main__.py:61`).

## The three subcommands

### `smoke` — offline mock race

`apps/arena/src/__main__.py:107` (`_cmd_smoke`). No API key required. Two mock profiles (`mock-a`, `mock-b`) run 10 turns each through the deterministic stub in `apps/arena/src/invoke.py:158` (`build_mock_invoke`). Used by CI (`just arena-smoke`) and as a 10-second sanity check after touching anything in `apps/arena/src/` or `packages/evaluation/src/`.

### `race` — head-to-head against real Claude

`apps/arena/src/__main__.py:89` (`_cmd_race`). Reads a YAML profile (default `apps/arena/src/profiles/v1.yaml`), builds one `ChatWire` per variant via `make_wire` (`apps/arena/src/invoke.py:124`), runs them concurrently via `asyncio.gather` (`apps/arena/src/race.py:43`, `_race_with_factory`), and prints a ranked table from `arena.metrics.summarise` (`apps/arena/src/metrics.py:76`). Each variant is one `ConfigProfile` — a frozen Pydantic model with `name / model / temperature / prompt_variant / wire / base_url` (`apps/arena/src/config_profile.py:29`). `wire` is where a race selects `zen` or `anthropic`; every profile in one race must name the same vendor, since they share one credential. Variants are isolated: they never share API state or any singleton from `apps/agent/src/`.

### `rank` — Bradley–Terry tournament

`apps/arena/src/__main__.py:143` (`_cmd_rank`). Runs `rounds × scenarios × profiles` race-instances, scores each final `WorldState` lexicographically (age → population → food+wood, `apps/arena/src/ranking.py:80`), builds a pairwise win matrix, and solves for Bradley–Terry log-ratings via iterative Minorization-Maximization (`apps/arena/src/ranking.py:104`, `_solve_bt`). 95% CIs come from percentile bootstrap with a configurable seed (`apps/arena/src/ranking.py:174`). Before kicking off, the CLI prints an estimated dollar cost (Haiku pricing) and prompts for confirmation when stdin is a TTY — see [Chapter 17 — Ranking Pipeline](./17-ranking-pipeline.md) for the BT math and the scenarios story.

## How the topology fits together

Every subcommand runs its producer through the same broker-shaped shim:

```
                ┌────────────────────────────────┐
                │ arena CLI (_run_through_broker)│
                └────────────┬───────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ make_broker()       │  ARENA_BROKER_BACKEND
                  │  → InProcess / Redis│  (default: inprocess)
                  └──────────┬──────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼─────┐        ┌─────▼─────┐
   │ race /  │ events  │ Multi-Run │ INSERT │ DuckDB    │
   │ rank /  │────────▶│ Broker    │───────▶│ file (per │
   │ smoke   │         │ Sink      │        │  command) │
   └─────────┘         └───────────┘        └───────────┘
        │                                          ▲
        │  (live consumers — Phase 2+)             │
        │  apps/api/src/server.py /events SSE ─────┘ (cold path)
        └─────────────────────────────────────────────►
```

The shim is `_run_through_broker` (`apps/arena/src/__main__.py:68`): it builds the broker via `evaluation.broker_factory.make_broker()`, opens one DuckDB connection, and wires both together through `MultiRunBrokerSink` (`packages/evaluation/src/duckdb_persister.py:99`). Producers (`race`, `rank`, `smoke`) emit events into the sink; the sink auto-opens a broker run on first sight of each `event.run_id` and spawns a per-run drainer that writes to DuckDB. `await sink.close_all()` at the end guarantees the file is consistent on disk before the CLI returns.

This shape means **the live web UI and the post-mortem DuckDB query are the same thing** — both subscribe to the broker's `Seq`-ordered stream, just at different times. The full motivation lives in [`docs/design/event-broker-architecture.md`](../design/event-broker-architecture.md) (now a frozen historical spec).

## What's *not* in the arena package

- **The agent's real-game loop** (`apps/agent/src/game_loop.py`) is separate — the arena imports `gameplay_agent.synth_game_loop` (`apps/arena/src/race.py:22`) which is a stripped-down loop that talks to `WorldState` instead of pyautogui. The real-game tier (Parts 1–4) never reaches into `apps/arena/src/`.
- **The detection server** (`apps/detection-server/src/`). Arena variants run with synthetic perception (`packages/evaluation/src/world_sim.render()`); no YOLO involved. See [Chapter 18](./18-synthetic-world-sim.md).
- **Score persistence across runs.** Ranking ratings *are* persisted as `MetricPayload` events under a synthetic `run_id="ranking"` (`apps/arena/src/ranking.py:235`, `_emit_ratings`), but cross-CLI-invocation aggregation is not built. Each `arena rank` invocation is self-contained.

## Profiles you can copy

- `apps/arena/src/profiles/v1.yaml` — two-way prompt comparison (`bare` vs `strategy`), Haiku, temperature 0.0, 60 turns. Cheapest interesting race.
- `apps/arena/src/profiles/ranking-v1.yaml` — same two profiles, temperature 0.5 for sampling variance, 5 rounds × default scenarios. Total cost ~$1.20.

The `ConfigProfile` Pydantic model (`apps/arena/src/config_profile.py:26`) and the `RaceConfig` / `RankingConfig` loaders (`apps/arena/src/config_profile.py:37` and `:53`) are the schema. Adding a new variant is a 4-line YAML addition, not a code change.

## Next reading

- [Chapter 15 — Event Broker](./15-event-broker.md) — what `make_broker()` returns and how the two impls differ.
- [Chapter 16 — DuckDB Persister and Replay](./16-duckdb-persister-and-replay.md) — the cold path: schema, `stream_cold`, `fork()`.
- [Chapter 17 — Ranking Pipeline](./17-ranking-pipeline.md) — Bradley–Terry, scenarios, bootstrap CIs.
- [Chapter 18 — Synthetic World Sim](./18-synthetic-world-sim.md) — the AoE2-lite economy model the arena races against.
