# `arena/` — Synthetic Arena CLI

The orchestration tier for evaluating the agent against an in-memory AoE2-lite world. No game, no VM, no screenshots. Three subcommands plus a YAML profile system, all wired through one event broker into one DuckDB log per run.

The FastAPI replay/fork web server lives in the sibling `arena-web` package
(deployed separately so the slim CLI install doesn't pull `fastapi`/`uvicorn`).

## What's here

```
packages/arena/src/
├── __main__.py        # CLI: race / smoke / rank
├── invoke.py          # Synth-arena invoke callables (real Claude + mock)
├── race.py            # async race controller (asyncio.gather over variants)
├── ranking.py         # Bradley–Terry pairwise ranking + bootstrap CIs
├── scenarios.py       # Named starting WorldState scenarios
├── config_profile.py  # Pydantic schema: ConfigProfile, RaceConfig, RankingConfig
├── metrics.py         # In-memory summary table from SynthLoopResult
├── prompts.py         # Prompt-variant registry (looked up by `prompt_variant`)
└── profiles/          # *.yaml profiles consumed by the CLI
```

## Common commands

```bash
# Offline smoke test, no API key, no spend
just arena-smoke

# Race profiles head-to-head (real Claude, ~$0.02–$0.20)
just arena-race                  # default packages/arena/src/profiles/v1.yaml
just arena-race path/to/v2.yaml  # custom

# Bradley-Terry tournament with 95% CIs (~$1.20 default)
just arena-rank                  # default packages/arena/src/profiles/ranking-v1.yaml

# Web UI for inspecting any past or live run
just arena-web-dev               # backend on :8000
just arena-ui-dev                # frontend on :5173 (separate terminal)
```

Every command writes its event log to `logs/arena/<YYYY-MM-DD>/<label>-<HHMMSS>.duckdb`.

## Where to read more

- [Chapter 14 — Arena Overview](../../docs/part6-evaluation-arena/14-arena-overview.md) — what each subcommand does, when to use which.
- [Chapter 17 — Ranking Pipeline](../../docs/part6-evaluation-arena/17-ranking-pipeline.md) — Bradley-Terry math, scoring, bootstrap.
- [Chapter 18 — Synthetic World Sim](../../docs/part6-evaluation-arena/18-synthetic-world-sim.md) — the AoE2-lite economy model races run against (`evaluation/world_sim.py`).
- [Chapter 15 — Event Broker](../../docs/part6-evaluation-arena/15-event-broker.md) — how every CLI flows through `make_broker()` + `MultiRunBrokerSink`.

## Conventions

- Profile YAML loaders are frozen Pydantic models (`config_profile.py`). Adding a new field means updating the model.
- Each variant gets its own `AsyncAnthropic` client — they never share API state or any singleton from `gameplay-agent`.
- `synth_game_loop` (in `gameplay-agent`) is the loop the arena drives, *not* the real `game_loop.py`. They share the same providers but the synth loop talks to `WorldState` instead of pyautogui.
