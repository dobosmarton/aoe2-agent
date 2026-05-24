# Chapter 17 — Ranking Pipeline

`python -m arena rank` runs a multi-round, multi-scenario tournament between profiles and produces Bradley–Terry log-ratings with 95% bootstrap confidence intervals. This chapter walks through the math, the YAML, and the failure modes.

## The shape of the output

```
Rank  Profile                    Rating  95% CI               Wins/Total
-------------------------------------------------------------------------
   1  strategy                  +0.412  [+0.180, +0.640]        17/20
   2  bare                      -0.412  [-0.640, -0.180]         3/20

Ranking: 2 profiles x 4 scenarios x 5 rounds x 60 turns = 20 race-instances
Estimated cost (Haiku pricing): ~$1.22
```

Ratings are log-ratings centred on zero (mean = 0). Two profiles separated by `r₁ - r₂` in log-rating units have an expected win probability of `1 / (1 + exp(r₂ - r₁))`. A +0.412 rating gap implies a ~60% expected win rate against the bottom of the table.

## The configuration

`packages/arena/src/config_profile.py:53` defines `RankingConfig`:

```yaml
turns: 60                  # turns per race-instance
rounds: 5                  # how many times each (profile × scenario) is replayed
profiles:                  # ≥2 ConfigProfiles (name / model / temperature / prompt_variant)
  - name: bare
    model: claude-haiku-4-5-20251001
    temperature: 0.5       # 0.5 gives sampling variance so rounds aren't degenerate
    prompt_variant: bare
  - name: strategy
    model: claude-haiku-4-5-20251001
    temperature: 0.5
    prompt_variant: strategy
scenarios: []              # empty = DEFAULT_SCENARIOS; otherwise list of names
bootstrap_samples: 1000    # percentile-bootstrap iterations for CIs
bootstrap_seed: 42         # makes CIs reproducible
```

`packages/arena/src/profiles/ranking-v1.yaml` is the shipped default. Cost depends on `rounds × |scenarios| × |profiles| × turns × per-turn-token-cost` — the CLI prints the estimate before running and prompts on TTY.

## The scoring function

Each completed race-instance produces a per-profile final `WorldState`. `composite_score` (`packages/arena/src/ranking.py:80`) computes a **lexicographic** tuple:

```python
(AGE_SEQUENCE.index(state.age), state.population, state.food + state.wood)
```

Higher beats lower at the first level that differs. Reaching Feudal Age dominates any amount of Dark Age economy; within an age, population dominates resource hoarding. The scoring is intentionally not a weighted sum — the agent should not be able to "cheat" by stockpiling food in Dark Age to outrank a Feudal opponent.

The scoring function is passed in as `score_fn` (`ranking.py:307`), so a research run with different goals can swap it without changing the harness.

## The Bradley–Terry solver

`_solve_bt` (`packages/arena/src/ranking.py:104`) implements iterative Minorization-Maximization on the pairwise win matrix. Reference: Hunter (2004), "MM algorithms for generalized Bradley–Terry models". Output is mean-centred log-ratings.

Two practical guardrails:

- **Symmetric `+0.5` smoothing.** `_BT_PRIOR = 0.5` (`ranking.py:101`) adds half a phantom win each way to every off-diagonal cell. Keeps the MLE bounded when one profile is undefeated (otherwise its rating diverges to `+inf`) and when all outcomes are ties (otherwise denominator = 0 → division crash). Same approach LMSys Chatbot Arena uses.
- **Convergence guard.** `_MM_MAX_ITERS = 1000`, tolerance `1e-6`. Raises `RankingError` on non-convergence rather than returning a half-fit. In practice the solver converges in tens of iterations on real workloads.

The function takes a square `int` win matrix where `wins[i, j]` is the count of times profile `i` beat profile `j`. The win matrix is built by `_wins_from_outcomes` (`ranking.py:151`): outcomes are grouped by `(round_idx, scenario_name)` so that comparisons only happen between profiles that played the same starting state in the same round.

## The bootstrap CI

`_bootstrap_ci` (`ranking.py:174`) is straight percentile bootstrap:

1. Resample `n_outcomes` outcomes with replacement.
2. Rebuild the win matrix from the resample.
3. Solve BT.
4. Repeat `bootstrap_samples` times (default 1000).
5. Take the 2.5th and 97.5th percentiles per profile.

The `bootstrap_seed` knob (`config_profile.py:68`) seeds `np.random.default_rng` so two `arena rank` runs over the same outcomes produce the same CI bounds. Degenerate bootstrap resamples (where one profile happens to be undefeated) raise `RankingError` from `_solve_bt` and are skipped — `ranking.py:191`.

## Scenarios

`packages/arena/src/scenarios.py` defines `DEFAULT_SCENARIOS` (`scenarios.py:88`): four named starting `WorldState` positions, deliberately covering a range of openings:

| Scenario | Food | Wood | Pop | Notes |
|---|---|---|---|---|
| `balanced` | 200 | 150 | 8 | Standard Dark Age start |
| `food-poor` | 80 | 200 | 8 | Forces early food prioritisation |
| `wood-poor` | 300 | 40 | 8 | Stresses the lumber-camp gate |
| `late-start` | 120 | 120 | 6 | Smaller pop cap, slower opener |

`Scenario` is a frozen dataclass (`scenarios.py:16`); adding a new one is a 1-tuple addition to `DEFAULT_SCENARIOS`. Specifying a partial list in `ranking-config.scenarios:` calls `get_scenario(name)` (`scenarios.py:96`) per entry — unknown names raise `KeyError` rather than silently dropping the scenario.

## How a `rank` invocation flows

```
arena/__main__.py:_cmd_rank
    └─ _run_through_broker(db_path, lambda sink: rank(config, api_key, sink))
            └─ arena/ranking.py:rank
                  └─ _rank_with_race_fn
                       ├─ _select_scenarios(config)         # scenarios.py
                       ├─ _collect_outcomes(...)            # rounds × scenarios loop
                       │    └─ race(race_config, state, sink)   # arena/race.py
                       │         └─ asyncio.gather over profiles
                       │              └─ synth_game_loop(...)   # gameplay_agent
                       ├─ _build_result(outcomes, names, ...)
                       │    ├─ _wins_from_outcomes
                       │    ├─ _solve_bt(wins)
                       │    └─ _bootstrap_ci(...)
                       └─ _emit_ratings(sink, ranking_id, result)  # MetricPayload events
```

Everything is recorded. Final ratings + CI bounds are emitted as `MetricPayload` events under a synthetic `run_id="ranking"` (`ranking.py:235`, `_emit_ratings`) with metric names `ranking_rating_<profile>`, `ranking_ci_lo_<profile>`, `ranking_ci_hi_<profile>`. Future cross-CLI aggregation could read those out of DuckDB; today nothing does.

## When the result is suspect

- **CIs that overlap zero or each other** — not enough rounds. Bump `rounds:` and rerun.
- **One profile undefeated, CI hits the smoothing bound** — `_BT_PRIOR` is preventing divergence; run more rounds or add a third profile so the win matrix isn't degenerate.
- **`RankingError: BT solver did not converge`** — extremely rare; means the win matrix has a structure the MM solver can't fit in 1000 iterations. Inspect the matrix manually (`_wins_from_outcomes_for_test` is re-exported at `ranking.py:341` for this).
- **One scenario with very different ratings than the others** — that's signal, not noise. `_collect_outcomes` keeps scenario as part of the matchup key, so per-scenario per-profile breakdowns are recoverable from the `MetricPayload` events for richer post-hoc analysis.

## Offline mode for tests

`rank_with_mock` (`ranking.py:323`) wires `race_with_mock` from `packages/arena/src/race.py:84` into the same `_rank_with_race_fn` pipeline. No API key needed, deterministic outcomes. Used by ranking unit tests to exercise the BT solver and bootstrap CI without spending dollars.

## Related reading

- [Chapter 14 — Arena Overview](./14-arena-overview.md) — where `rank` fits among the three CLI subcommands.
- [Chapter 18 — Synthetic World Sim](./18-synthetic-world-sim.md) — the world the scenarios start from.
- [ADR 0004 — Bradley–Terry Ranking](../adr/0004-bradley-terry-ranking.md) — why BT over simple win-rate.
