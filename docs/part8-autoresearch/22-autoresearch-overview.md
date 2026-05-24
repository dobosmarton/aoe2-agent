# Chapter 22 — Autoresearch Overview

`autoresearch/` is the **prompt-optimization loop**: an LLM proposes a small targeted edit to `prompts/system.md`, the agent plays a real game with the modified prompt, a composite score decides accept or revert, and the change is committed to git so the history is auditable.

The pattern is borrowed from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): modify → evaluate → keep/revert → repeat. The original 5-phase plan lives at [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) (frozen historical spec). This chapter covers what's actually shipped: Phase 0 (foundation) and Phase 1 (prompt-mutation loop).

## How a single experiment runs

`autoresearch/orchestrator.py:73` — `Orchestrator.run_experiment`:

1. **Read recent experiments and failure modes.** `get_recent_experiments(5)` from the TSV ledger; `_extract_failure_modes` derives natural-language hints from the last game's metric breakdown (e.g. "Population stayed very low — agent may not be queueing villagers", `orchestrator.py:254`).
2. **Propose a change.** `PromptMutator.propose_change` (`prompt_mutator.py:61`) sends the current prompt, recent results, and failure modes to Haiku. The mutator system prompt (`prompt_mutator.py:20`) enforces tight constraints: change ≤5 lines, don't touch `## Output Format` or `## Game State Detection`, focus on one specific weakness, return JSON with `description`, `old_text`, `new_text`, `rationale`.
3. **Apply** via string `.replace(old_text, new_text, 1)`. Refuses to apply if `old_text` overlaps a protected section.
4. **Commit** to git as `[autoresearch] <experiment_id>: <description>` so every change is recoverable.
5. **Run a game** via `autoresearch.game_runner.run_game` with the modified prompt. Captures the full `AgentMemory`, computes a `GameScore`.
6. **Accept or revert.** Accept if `score.composite >= self.best_score - epsilon` (default `epsilon = 0.02`). On revert, `git checkout -- prompts/system.md` followed by a `[autoresearch] revert: prompt change rejected` commit.
7. **Log** to the experiment ledger TSV.

A baseline game with the unmodified prompt runs once before the loop starts (unless `--no-baseline`) so the first comparison has a number to beat.

## Scoring

`autoresearch/metrics.py` (`compute_score`) is a weighted composite, with weights frozen in `autoresearch/config.yaml`:

| Component | Weight | Source |
|---|---|---|
| Survival time | 0.30 | `metrics.survival_time / 1200s` |
| Peak population | 0.25 | `metrics.peak_population / 50` |
| Age advancement | 0.20 | `metrics.highest_age` mapped to ordinal |
| Economy (food) | 0.15 | `metrics.food_gathered / 5000` |
| Action success rate | 0.10 | fraction of actions with `state_changed=True` |

All five sub-scores are clamped to `[0, 1]` then combined. The `composite` field is what accept/reject decisions use.

The weight choice is an explicit value judgment: survival dominates because games that die early have no signal on the other axes, and population beats age because the agent learns to stockpile faster than it learns to advance.

## Cross-game memory chain

`autoresearch/memory_chain.py` is the second half of the loop. After each game, `MemoryChain.extract_memories` (`memory_chain.py:107`) sends a turn-by-turn summary to Haiku with an extraction prompt that demands first-person imperative rules ("I should…", not "the agent should…") and one rule per file.

The output is markdown frontmatter + body, written under `memories/NNN_<title>.md`:

```markdown
---
type: strategy
title: stop_villagers_at_low_food
game_id: exp_0042
applies_when: food < 500 AND age == "Dark Age"
score_impact: negative
created: 2026-05-24T14:21:09+00:00
---

I should stop queueing villagers when food drops below 500 in Dark Age.
Last game, I queued three villagers between turn 14 and 18 which delayed
my age-up by 4 turns.
```

The memory loader (`MemoryChain.load_memories`) ranks these for the next game's context: negative impact first (traps to avoid), then positive (patterns to repeat), then neutral; within each tier newest-first. A token budget caps the loaded text at ~800 tokens.

The dedup is intentionally minimal — same-title memories are skipped (`memory_chain.py:152`) but no semantic dedup. The memory dir is meant to be human-reviewable; delete bad ones manually.

## Entry points

| Command | What it does |
|---|---|
| `python -m autoresearch.orchestrator [--max-experiments N] [--time-budget 1200]` | Run the prompt loop. Prompts you to start an AoE2 game between experiments. |
| `python -m autoresearch.game_runner [--time-budget 1200] [--description "..."]` | One-off game with metrics + memory extraction, logged to the ledger as a manual entry. |
| `python -m autoresearch.game_runner --max-iterations 50` | Useful for shorter test runs. |

Both spawn the same `gameplay_agent.game_loop` — autoresearch is a thin wrapper over the real-game tier, not a separate runtime. The agent plays a real AoE2 game on the Windows VM; autoresearch is what's running on the host watching the result.

## What's *not* shipped

Phases 2–5 of the original plan (context tuning, strategy mining, automated game restart, detection active learning, training pipeline improvements) — see the frozen [autoresearch-plan.md](../design/autoresearch-plan.md). The `config.yaml` has `context_loop.enabled: false` and `strategy_loop.enabled: false` placeholders to signal the intended extension points.

## When to use this vs `arena rank`

| | Arena rank | Autoresearch |
|---|---|---|
| **What it changes** | A static set of profile variants you authored | The prompt itself, evolved over runs |
| **What it runs against** | Synthetic AoE2-lite world | Real AoE2 on the Windows VM |
| **Cost per experiment** | ~$0.01 per variant per round | $1–5 per game (20-minute Haiku run) |
| **Statistical guarantee** | Bradley–Terry CIs at 95% | Single-sample greedy accept/reject |
| **Time per experiment** | Seconds | 20+ minutes (a full game) |
| **Auditability** | DuckDB event log | Git commits + TSV ledger + memory files |

These are complementary, not interchangeable. Use `arena rank` to pick between hand-crafted variants cheaply; use autoresearch to evolve a single prompt against real-game evidence expensively.

## Related reading

- [Chapter 23 — Prompt Mutation and Memory](./23-prompt-mutation-and-memory.md) — the mutator and memory-chain internals.
- [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) — the original 5-phase plan (frozen historical).
- [Chapter 17 — Ranking Pipeline](../part6-evaluation-arena/17-ranking-pipeline.md) — the synthetic-tier counterpart.
