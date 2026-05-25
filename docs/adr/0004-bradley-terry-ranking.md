# ADR 0004 — Bradley–Terry ranking over simple win-rate

**Status:** Accepted (2026-05). Shipped as Phase 8.
**Context:** [Chapter 17 — Ranking Pipeline](../part6-evaluation-arena/17-ranking-pipeline.md).

## Decision

For multi-profile evaluation, fit a Bradley–Terry MLE on the pairwise win matrix via iterative Minorization-Maximization. Report 95% percentile-bootstrap confidence intervals. Use a symmetric `+0.5` smoothing prior to keep the MLE bounded when one profile is undefeated.

## What we considered

| Option | Pros | Cons |
|---|---|---|
| **Simple win-rate** (wins / total) | Trivial to compute. | Doesn't account for matchup structure when not all profiles meet equally; no statistical guarantee. |
| **Elo iterative update** | Familiar from chess. | Sensitive to scheduling; no natural confidence intervals; tunable K-factor adds another knob. |
| **Bradley–Terry MLE** | Closed-form fit; natural pairwise interpretation; CI via bootstrap. | Needs smoothing to handle undefeated profiles; requires `numpy`. |
| **TrueSkill / Glicko** | Per-player rating with uncertainty built in. | Heavy machinery for our small-N case; extra deps. |

## Why Bradley–Terry

- **The matchup structure is already pairwise.** A "race instance" pits N profiles against each other on one scenario. The natural primitive is the win matrix.
- **Same approach LMSys Chatbot Arena uses.** Well-understood, widely-explained model; reviewers don't need to learn a custom rating scheme.
- **Closed-form symmetric smoothing handles degenerate cases.** When profile A beats profile B in every round, vanilla MLE diverges to `+inf`. The `+0.5` phantom-games prior (`apps/arena/src/ranking.py:101`) keeps ratings finite; bootstrap CIs reflect the resulting uncertainty.
- **Bootstrap CI gives reviewers what they actually need:** "Is the difference between strategy and bare significant?" — answered by whether the 95% CIs overlap.

## What we explicitly traded away

- **Cross-round score persistence.** Each `arena rank` invocation is self-contained. There's no global rating ladder accumulating across CLI runs. The `MetricPayload` events emitted under `run_id="ranking"` (`apps/arena/src/ranking.py:235`) make this possible later, but nothing reads them today.
- **Per-scenario ratings.** `_collect_outcomes` keeps scenario in the matchup key (so cross-scenario pairs don't compare), but the final rating is aggregated across scenarios. Per-scenario breakdowns are recoverable from the events; the table summary collapses them.

## Why the lexicographic scoring (not a weighted sum)

`composite_score = (age_index, population, food + wood)` (`ranking.py:80`). Lexicographic, not weighted.

The rationale is the same one that drove the autoresearch composite score: reaching a higher age dominates any amount of lower-age economy. A weighted sum would let an agent "cheat" by stockpiling food in Dark Age to outrank a Feudal-Age opponent. Lexicographic ordering encodes the actual game-strategic intuition: you cannot trade age progress for resources.

The score function is passed in (`ranking.py:307` — `score_fn` parameter) so research with different goals can swap it without changing the harness.

## Consequences

**Positive**
- Statistically defensible head-to-head comparisons with low ceremony.
- Pluggable scoring (`score_fn`) keeps the harness general.
- Bootstrap with `bootstrap_seed` keeps CIs reproducible across runs.

**Negative**
- Needs `numpy` (already a dep — for `world_sim`).
- The MM solver caps at 1000 iterations; in pathological cases a non-converged result raises `RankingError`. Rare in practice.
- Bootstrap is N²-ish in samples × profiles² for win-matrix construction; at default `bootstrap_samples=1000` and small N this is negligible. Larger evaluations might need vectorization.

## Related

- [Chapter 17 — Ranking Pipeline](../part6-evaluation-arena/17-ranking-pipeline.md) — operating guide.
- Hunter (2004), "MM algorithms for generalized Bradley–Terry models" — the solver reference.
