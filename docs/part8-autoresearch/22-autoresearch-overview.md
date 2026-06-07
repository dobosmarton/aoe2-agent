# Chapter 22 — Autoresearch Overview

`apps/autoresearch/src/` is the **prompt-optimization loop**: an LLM proposes several small targeted edits to `prompts/system.md`, the candidates race in a **successive-halving tournament** of real games, and the winning edit is kept (committed to git) only if its mean score beats the current baseline. Every trial and the final decision are logged so the history is auditable.

The pattern is borrowed from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): modify → evaluate → keep/revert → repeat. The original 5-phase plan lives at [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) (frozen historical spec). This chapter covers what's actually shipped: Phase 0 (foundation) and Phase 1 (prompt-mutation loop).

<aside class="prereqs">

[Chapter 5 — Prompt Engineering](../part2-llm-integration/05-prompt-engineering.md) for the prompt this loop mutates. No prior background in prompt-optimization needed — the chapter introduces the [Karpathy autoresearch lineage](../glossary.md#o) in place.

</aside>

## How a tournament runs

`apps/autoresearch/src/orchestrator.py` — `Orchestrator.run_tournament`:

1. **Read recent experiments and failure modes.** `get_recent_experiments(5)` from the TSV ledger; `_extract_failure_modes` derives natural-language hints from the last game's metric breakdown (e.g. "Population stayed very low — agent may not be queueing villagers").
2. **Propose N candidates.** `PromptMutator.propose_changes(..., n)` (`prompt_mutator.py:61`) asks Haiku for N **distinct** edits as a JSON array, each `{description, old_text, new_text, rationale}`. The mutator system prompt (`prompt_mutator.py:20`) enforces tight constraints: change ≤5 lines, don't touch `## Output Format` or `## Game State Detection`, one specific weakness per edit. Candidates whose `old_text` isn't found verbatim in the current prompt are filtered out before any game.
3. **Race them via successive halving** (`_run_halving_rounds`). Each round plays every surviving candidate **one game** — apply → commit → `run_game` → `git checkout` revert, so the next candidate starts from the same baseline — records its composite score, then keeps the top `keep_fraction` by **mean** composite and doubles down on the survivors. Games are human-started and sequential, capped by `games_budget`.
4. **Accept the winner** (the survivor with the best mean) only if `mean >= self.best_score - epsilon` (default `epsilon = 0.02`). On accept it is re-applied and committed as `[autoresearch] <id>: <description>`; otherwise the prompt stays at baseline (every trial already reverted).
5. **Log** each trial game and the final decision to the experiment ledger TSV (with the `tournament_id`, `candidate_id`, `round`, `game_in_round` columns).

No separate baseline game is needed: `best_score` comes from the ledger, and on a fresh ledger (0.0) the first tournament simply keeps its best candidate.

> **Cost note.** With the defaults (`n_candidates=3`, `halving_rounds=2`, `keep_fraction=0.5`), a tournament is roughly **5 supervised games per accepted change** versus 1 for the old greedy loop — the deliberate price of noise-robust selection.

<aside class="concept" data-title="Git as audit trail for ML experiments">

Most ML experiment tracking is via MLflow, Weights & Biases, or a custom database. We use git plus a TSV ledger plus markdown memory files. The choice is deliberate.

**What git buys us that a DB doesn't:** every prompt change is its own commit with a descriptive message; `git log prompts/system.md` shows every accepted change in chronological order; `git blame` tells you which experiment introduced a given line; `git revert <sha>` is a tested, well-understood escape hatch; and the diff between two prompts is just `git diff sha1 sha2`. None of these need a UI to interpret — they work with the tools every developer already knows.

**What the TSV ledger adds:** structured per-experiment metadata (score, accept/reject, latency, error) that you don't want in a commit message. Designed to be `awk`-able and `column -t`-readable from the terminal.

**What the memory files add:** human-readable rules that are versioned alongside the prompt itself (since `memories/*.md` lives in the repo). Bad rules are `rm`-able; a `git diff memories/` shows what the loop learned this week.

**Why this is the right shape *here*:** we run dozens of experiments, not millions. Auditability and post-hoc reproducibility matter more than fast queries over a huge search space. If we were running a million experiments per day, MLflow's parameter-search UIs would dominate. At our scale, the marginal value of "everything is in git" outweighs everything else.

</aside>

## Scoring

`apps/autoresearch/src/metrics.py` (`compute_score`) is a weighted composite, with weights frozen in `apps/autoresearch/src/config.yaml`:

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

`apps/autoresearch/src/memory_chain.py` is the second half of the loop. After each game, `MemoryChain.extract_memories` (`memory_chain.py:107`) sends a turn-by-turn summary to Haiku with an extraction prompt that demands first-person imperative rules ("I should…", not "the agent should…") and one rule per file.

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
| `python -m autoresearch.orchestrator [--max-experiments N] [--time-budget 1200] [--n-candidates 3] [--halving-rounds 2] [--keep-fraction 0.5] [--games-budget 6]` | Run the successive-halving tournament. Prompts you to start an AoE2 game before each trial. |
| `python -m autoresearch.game_runner [--time-budget 1200] [--description "..."]` | One-off game with metrics + memory extraction, logged to the ledger as a manual entry. |
| `python -m autoresearch.game_runner --max-iterations 50` | Useful for shorter test runs. |

Both spawn the same `gameplay_agent.game_loop` — autoresearch is a thin wrapper over the real-game tier, not a separate runtime. The agent plays a real AoE2 game on the Windows VM; autoresearch is what's running on the host watching the result.

<details class="deep-dive">
<summary>Deep dive — The prompt-optimization landscape (DSPy, OPRO, evolutionary strategies)</summary>

Our loop — *mutate the prompt → evaluate → keep or revert* — is one point in a broader space of automated prompt-optimization techniques. Knowing the neighbours helps you reach for the right tool and understand what the academic literature is talking about.

**OPRO** (*Optimization by PROmpting*, Yang et al., DeepMind 2023). One LLM is an "optimizer" that proposes new prompts conditioned on a history of `(prompt, score)` pairs; another LLM is the "scorer" that evaluates each proposal on a held-out task. Iterate until scores plateau. Effectively LLM-as-numerical-optimizer for the prompt search space — clever but assumes you can score quickly and cheaply (one LLM call per evaluation), which doesn't fit our setup where each evaluation is a 20-minute AoE2 game.

**DSPy** (Khattab et al., Stanford 2023). A *programming framework* in which you write your task as a typed pipeline of modules (`ChainOfThought`, `RetrieveAndRead`, etc.) and a compiler turns it into an optimized prompt via *demonstrations* — DSPy finds the best few-shot examples to inject by enumerating training data. The optimization target is the few-shot block, not the whole instruction prompt. Fits well when you have a labeled dataset and a quick way to score; less useful for long-horizon agent loops where the "trajectory" matters more than the per-call output.

**Evolutionary / genetic algorithms.** Treat the prompt as a chromosome, mutation operators as small edits (rewrite a sentence, swap two paragraphs, paraphrase a constraint), selection by score. Promptbreeder (Fernando et al., DeepMind 2023) is the best-known example: it co-evolves both the prompt and the mutation operators themselves. Powerful but operates on a population (typically 30+ candidates per generation), which is incompatible with our "one expensive evaluation at a time" budget.

**Karpathy's autoresearch** (our lineage). A single-candidate hill-climb: propose *one* mutation, evaluate it once, accept if it beats the current best by ε. No population, no statistical hypothesis testing. We **started** here, then extended it to a **successive-halving tournament** (propose N candidates, race them over rounds, keep the best by mean score) to buy noise-robustness and a little exploration without a population-sized blow-up in game count. Still auditable in git, still one expensive evaluation at a time per candidate.

**Reinforcement learning from outcome reward.** The full general form — treat each prompt as an action, each game's score as a reward, train a policy. Massive sample complexity, almost never practical for prompt-level optimization unless you can simulate evaluations cheaply.

**Where ours sits.** Successive halving over a small candidate pool is a middle ground: more exploration and noise-robustness than greedy hill-climb, far fewer evaluations than the OPRO/DSPy/Promptbreeder populations. The trade-off still hinges on ε (the accept threshold) being honestly calibrated to the noise floor of your evaluator — see [Chapter 18's "Determinism in LLM agents" deep dive](../part6-evaluation-arena/18-synthetic-world-sim.md) for why that's a load-bearing assumption.

**The right next step from here** depends on what runs cheaper for you: if evaluation is cheap, move toward OPRO/DSPy. If evaluation is expensive but each game produces rich per-turn data (as ours does), move toward strategy-mining the failure modes (Phase 2 of our original plan).

</details>

## What's *not* shipped

Phases 2–5 of the original plan (context tuning, strategy mining, automated game restart, detection active learning, training pipeline improvements) — see the frozen [autoresearch-plan.md](../design/autoresearch-plan.md). The `config.yaml` has `context_loop.enabled: false` and `strategy_loop.enabled: false` placeholders to signal the intended extension points.

## When to use this vs `arena rank`

| | Arena rank | Autoresearch |
|---|---|---|
| **What it changes** | A static set of profile variants you authored | The prompt itself, evolved over runs |
| **What it runs against** | Synthetic AoE2-lite world | Real AoE2 on the Windows VM |
| **Cost per experiment** | ~$0.01 per variant per round | $1–5 per game (20-minute Haiku run) |
| **Statistical guarantee** | Bradley–Terry CIs at 95% | Successive-halving over N candidates (mean of ≥1 game each) |
| **Time per experiment** | Seconds | 20+ minutes (a full game) |
| **Auditability** | DuckDB event log | Git commits + TSV ledger + memory files |

These are complementary, not interchangeable. Use `arena rank` to pick between hand-crafted variants cheaply; use autoresearch to evolve a single prompt against real-game evidence expensively.

<details class="deep-dive">
<summary>Deep dive — Multi-armed bandits and Thompson sampling (the smarter loop we haven't built yet)</summary>

Phase 2 of the original plan (frozen at `docs/design/autoresearch-plan.md`) would let the loop A/B test *context variants* — same prompt, different context strategies, decided on the fly. That's the canonical home of a **multi-armed bandit** algorithm, and worth understanding even before it ships because the same machinery applies to "which prompt variant should I evaluate next?"

**The bandit problem.** You have K arms (prompt variants). Each arm has an unknown payoff distribution. You can pull one arm per round and observe its reward. Goal: maximize total reward over N rounds. The tension: every round spent learning about a low-payoff arm is a round not earning from the high-payoff arm (**exploration vs exploitation**).

**The three canonical algorithms:**

- **ε-greedy.** With probability ε pick a random arm, otherwise pick the arm with the highest empirical mean. Trivial to implement. The downside: ε is fixed, so you keep exploring even after you have enough data; or you set it small and converge prematurely on a noisy local optimum.
- **UCB (Upper Confidence Bound).** Pick the arm that maximizes `mean + √(2·log(t) / n_arm)`. Provably optimal regret bounds, no hyperparameter to tune, deterministic. The downside: assumes bounded rewards and i.i.d. pulls — both assumptions fail with LLM agents whose scores drift as the game world drifts.
- **Thompson sampling.** Maintain a posterior distribution over each arm's mean (Beta–Bernoulli for binary, Normal–Inverse–Gamma for continuous). Each round, *sample* one number from each arm's posterior and pick the arm with the largest sample. This is the elegant Bayesian answer: arms with wide posteriors get sampled often (exploration), arms with tight high-mean posteriors get picked often (exploitation), and the trade-off self-tunes from the data with no hyperparameter.

**Why Thompson is usually the right choice for prompt evaluation.** Empirically it matches or beats UCB on most real bandits, the posterior gives you natural per-arm uncertainty (you can stop early when one arm dominates), and it composes cleanly with contextual features (Bayesian linear regression posterior → contextual Thompson sampling).

**The catch for our use case:** classic bandits assume rewards are i.i.d. — pull arm K, get reward `r ~ D_K`. Our "reward" is a 20-minute game whose outcome depends on the prompt, the game's RNG, the LLM's nondeterminism (see [Chapter 18 §Determinism deep dive](../part6-evaluation-arena/18-synthetic-world-sim.md)), and small drift in the world simulator. Naive Thompson treats this as just noisier rewards and works *fine*; better is to model the drift explicitly (a state-space model on top of the posterior). We're not there yet.

**The simplest useful first version** — when Phase 2 ships — is probably ε-greedy with ε=0.2 for the first 50 experiments, then Thompson once you have enough data to fit Beta priors. Don't reach for contextual bandits or hierarchical models until you've felt the limit of the simpler version.

</details>

## Related reading

- [Chapter 23 — Prompt Mutation and Memory](./23-prompt-mutation-and-memory.md) — the mutator and memory-chain internals.
- [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) — the original 5-phase plan (frozen historical).
- [Chapter 17 — Ranking Pipeline](../part6-evaluation-arena/17-ranking-pipeline.md) — the synthetic-tier counterpart.
