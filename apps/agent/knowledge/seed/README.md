# Seed knowledge — frozen 2026-08-17

Everything the agent knows today, lifted out of code comments, prompt strings and
old memory files **before** `ADAPTIVE-AGENT-PLAN.md` Phase 1 deletes
`reactive.py`.

Phase 0.1 of the plan. Nothing here changes behaviour; it is a capture.

## Contents

| Path | What |
| --- | --- |
| `findings.md` | 48 run-review findings. **36 live only in source** that Phase 1 rewrites. |
| `rules/dark_age.yaml` | The Dark Age rules from `reactive.py`, one per constant. |
| `rules/feudal_age.yaml` | The Feudal rules. |
| `rules/allocation.yaml` | Gather ratios, famine overrides, wood/gold bank targets, idle sizing. |
| `rules/safety_floor.yaml` | 10 code-enforced invariants. Game facts, never strategy. |
| `prompt_rules.md` | The strategy in `core.md` and `prompts/ages/*.md`. |
| `archive/` | The 9 memory files the agent ever wrote, plus conversion notes. |
| `baseline.tsv` | The 14 ledger rows, stamped `score_version: 1`. |

## Rules of use

**Seed rules are protected.** Per plan 0.2, a rule carrying `provenance: seed/*`
is exempt from retirement until it has been ablated at least once. The first
noisy sweep must not delete knowledge that took months to acquire.

**The safety floor is never reranked and never retired** (plan 6.1).

**Weights here are hand-seeded**, chosen to reproduce the group order in
`reactive.decide()`. Phase 6.1 replaces them with ablation results.

## Three things this capture turned up

**1. The memory system already found 4 of the hardcoded rules.** Months before a
human wrote them into Python, `memory_chain` had written notes proposing the
housing threshold, the age-up push, the h/z failure and the build-retry cap. It
lacked a way to act on them and a way to measure them. See `archive/README.md`.
This is the strongest available evidence that the plan's seeded arm is worth
running.

**2. One rule, four different thresholds.** Build a house at `pop_cap - 3`
(`core.md`), at `4/5 pop` (`game_knowledge.py`), rejected above 4 headroom
(`executor.py`), triggered at 2 (`reactive.py`). The registry collapses them to
one.

**3. Deduplication by title does not work.** 5 of the 9 archived memories say the
same thing about housing, all with different titles. Plan 4.2 replaces title
matching with trigger overlap; these files are the evidence for it.

## Known gaps

`prompt_rules.md` ends with 7 behaviours that exist in **no** code path — the
proactive-farm rule, the entire Castle Age up-gate, permanent gold villagers,
farm reseeding, and all technology research. The seed registry does not
reproduce them. They are the first candidates for strategist-authored rules.

## What this is for

Plan 6.3 runs two arms:

- **COLD** — `rules/safety_floor.yaml` only.
- **SEEDED** — the safety floor plus everything else here.

The gap between them measures what this knowledge was worth.
