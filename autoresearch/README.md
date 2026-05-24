# `autoresearch/` — Prompt Optimization Loop

LLM-driven evolution of `prompts/system.md` against real-game evidence. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): modify → evaluate → keep/revert → repeat.

## What's here

```
autoresearch/
├── orchestrator.py        # Main loop: mutate → run → score → accept/revert → log
├── prompt_mutator.py      # LLM-driven prompt edits with protected-section guards
├── memory_chain.py        # Cross-game memory: extract rules after each game, load before next
├── game_runner.py         # Wraps gameplay_agent.game_loop, adds metrics + memory extraction
├── metrics.py             # Composite GameScore (survival/pop/age/economy/action-success)
├── experiment_log.py      # TSV ledger at experiments/results.tsv
├── json_utils.py          # extract_json_object — tolerant LLM-output parser
└── config.yaml            # Scoring weights + loop toggles
```

## Common commands

```bash
# One-off manual game with metrics + memory extraction (logged as a manual entry)
python -m autoresearch.game_runner --time-budget 600 --description "baseline"

# Full prompt loop (prompts you to start each game manually)
python -m autoresearch.orchestrator --max-experiments 5 --time-budget 1200

# Same loop, no baseline game
python -m autoresearch.orchestrator --no-baseline
```

Each accepted change is a git commit on the main repo (`[autoresearch] <id>: <description>`). Each rejected change is reverted via `git checkout -- prompts/system.md` and committed as `[autoresearch] revert: prompt change rejected`. Run history is in `experiments/results.tsv` (not source-controlled; per-machine state).

## Reading order

- [Chapter 22 — Autoresearch Overview](../docs/part8-autoresearch/22-autoresearch-overview.md) — the loop and the scoring composite.
- [Chapter 23 — Prompt Mutation and Memory](../docs/part8-autoresearch/23-prompt-mutation-and-memory.md) — mutator + memory-chain internals.
- [`docs/design/autoresearch-plan.md`](../docs/design/autoresearch-plan.md) — the original 5-phase plan (frozen historical; only Phases 0–1 are shipped).

## When to use this vs `arena rank`

| | Arena rank | Autoresearch |
|---|---|---|
| What it changes | A static set of profile variants you authored | The prompt itself, evolved over runs |
| Runs against | Synthetic AoE2-lite world | Real AoE2 on the Windows VM |
| Cost / experiment | ~$0.01 / variant / round | $1–$5 (20-min game) |
| Statistical guarantee | Bradley-Terry CIs at 95% | Single-sample greedy accept/reject |
| Auditability | DuckDB event log | Git commits + TSV ledger + memory files |

Complementary, not interchangeable.

## Notes for contributors

- The mutator has hard constraints baked into its system prompt — don't loosen them without thinking about reversibility. The `PROTECTED_SECTIONS` list in `prompt_mutator.py` is the second line of defence.
- `memories/*.md` files are human-reviewable by design. If a memory is bad advice, delete the file.
- `EXTRACTION_SYSTEM` in `memory_chain.py` demands first-person imperative voice — if you tweak it, keep that property; the agent's downstream context assembly assumes it.
