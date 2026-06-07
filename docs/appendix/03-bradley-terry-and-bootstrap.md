# Appendix C — Bradley-Terry MLE and bootstrap confidence intervals

This appendix is the reference behind [Chapter 17 (Ranking Pipeline)](../part6-evaluation-arena/17-ranking-pipeline.md) and [ADR 0004 (Bradley-Terry Ranking)](../adr/0004-bradley-terry-ranking.md). The chapter shows how we *use* Bradley-Terry; this appendix explains *what it actually is, why it beats naive win-rate, how the math turns "A beat B" outcomes into a single skill number per agent, and why we wrap the result in bootstrap confidence intervals*.

You don't need a stats background to read this. You do need to be comfortable with the idea that "the model is more likely to be correct if it makes the data more probable" (the maximum-likelihood principle).

---

## C.1 The problem: ranking from pairwise comparisons

You ran 200 games. Different prompt variants of the agent played different scenarios against each other. You have a list of pairwise outcomes:

```
prompt_v3 beat prompt_v1
prompt_v1 beat prompt_v2
prompt_v3 beat prompt_v2
prompt_v2 beat prompt_v3
...
```

You want a **single skill rating** per prompt variant so you can rank them.

### Why win-rate is misleading

The naive answer is: `skill(p) = wins(p) / total_games(p)`. Three problems:

1. **Strength of schedule isn't accounted for.** A prompt that beat strong opponents at a 60% rate is much better than one that beat weak opponents at 80%.
2. **Sparse opponents are hard to compare.** Prompt A played only opponent X. Prompt B played only opponent Y. A vs B win-rates aren't on the same scale.
3. **Confidence isn't represented.** 6/10 wins and 60/100 wins both give 60% win-rate, but the second is far more meaningful.

Bradley-Terry fixes (1) and (2). Bootstrap CIs fix (3).

---

## C.2 The Bradley-Terry model

The model says: each competitor `i` has a hidden positive skill score `π_i`. The probability that `i` beats `j` in any given match is:

$$ P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j} $$

That's the whole model. Three things to note:

- **Only ratios matter.** Multiply all `π_i` by 10 and every probability is unchanged. So we fix the scale arbitrarily — typically `π_1 = 1.0`, or normalize so they sum to N. This is why people often report the *log* of `π` as the "skill," which is what makes Bradley-Terry skills look like Elo ratings (Elo is essentially Bradley-Terry on a log scale).
- **A single parameter per competitor.** No interaction terms — `P(i beats j)` is fully determined by `π_i` and `π_j`. This is a *strong* assumption (more on this in §C.6).
- **Ties aren't modeled.** Bradley-Terry assumes every match has a winner. Variants (Rao-Kupper, Davidson) handle draws explicitly. In our arena every match has a definite winner (highest score wins), so we don't need them.

### Why this captures "strength of schedule"

If `π_strong = 4` and `π_weak = 1`, then `P(strong beats weak) = 4/5 = 80%`. A new competitor `c` that beats `strong` 50% of the time must have `π_c = π_strong = 4`. A different competitor `c'` that beats `weak` 80% of the time has `π_c' / (π_c' + 1) = 0.8`, so `π_c' = 4`. **Same skill, despite very different win-rates against very different opponents.** That's the whole point.

---

## C.3 Fitting the model: maximum-likelihood

Given outcomes `D = {(i_k, j_k, winner_k)}`, the **likelihood** of observing this data under skills `π` is:

$$ L(\pi) = \prod_{k} P(\text{winner}_k \text{ beats loser}_k) = \prod_{k} \frac{\pi_{w_k}}{\pi_{w_k} + \pi_{l_k}} $$

Taking the log to turn the product into a sum:

$$ \ell(\pi) = \sum_{k} \left[ \log \pi_{w_k} - \log(\pi_{w_k} + \pi_{l_k}) \right] $$

The MLE is the `π` that maximizes `ℓ`. There's no closed-form solution, but the function is concave (in a transformed space), so gradient ascent converges.

### The MM algorithm (Hunter 2004)

In practice we don't use gradient ascent — we use the **Minorization–Maximization (MM) algorithm**, which has a beautifully simple per-iteration update:

$$ \pi_i^{(t+1)} = \frac{W_i}{\sum_{j \neq i} \frac{N_{ij}}{\pi_i^{(t)} + \pi_j^{(t)}}} $$

where `W_i` is the number of wins for `i`, and `N_{ij}` is the number of matches between `i` and `j`. Iterate, renormalize, repeat until `π` stops moving.

Why MM is nicer than gradient descent for this problem:

- No learning rate to tune. The update is closed-form per iteration.
- Monotonically increases the log-likelihood every step (proved in Hunter 2004).
- Each iteration is `O(N²)` for N competitors, which is fine until N gets into the thousands.

---

## C.4 The undefeated-profile problem (and the +0.5 fix)

A subtle pathology: if some competitor has *won every match they played*, the likelihood is maximized by `π_i → ∞`. The MLE doesn't exist (or, equivalently, is at infinity). Symmetric problem if a competitor lost every match: `π_i → 0`. The MM iteration in either case diverges or stalls.

Real arena runs hit this *all the time* — a brand-new prompt variant that wins its first 3 trial games has an "undefeated profile" until more data arrives.

The standard fix is **smoothing**: add a *phantom* half-win and half-loss between every pair of competitors. Effectively, we compute on a slightly modified outcome list:

```
real:   prompt_v3 beat prompt_v1 (× 5 times)
+phantom: prompt_v3 beat prompt_v1 (× 0.5)
+phantom: prompt_v1 beat prompt_v3 (× 0.5)
```

The phantom matches contribute 0.5 wins and 0.5 losses to each side of every pair, which keeps `π_i` finite. The bias introduced is small (and shrinks as real data grows) and is the standard accepted price for stability.

In our code this is the `+0.5 smoothing` parameter passed to the ranking solver in `apps/arena/src/ranking.py`.

---

## C.5 Bootstrap confidence intervals: "is this difference real?"

Fitting Bradley-Terry gives you a point estimate per agent. But the rank order from 50 games isn't going to be the same as the rank order from 500 games — there's sampling noise. **Bootstrap** is a non-parametric way to estimate that noise without making distributional assumptions.

### The recipe

```python
def bootstrap_ranking(matches, n_resamples=1000):
    n = len(matches)
    rankings = []
    for _ in range(n_resamples):
        resample = random.choices(matches, k=n)  # with replacement
        skills = bradley_terry_fit(resample)
        rankings.append(skills)
    return rankings
```

Then for each agent, the 2.5th and 97.5th percentiles of its skill across the 1000 bootstrap runs are your 95% confidence interval.

### Why this works

The bootstrap principle is: *the empirical distribution is an estimate of the true distribution. Resampling from it with replacement simulates drawing new samples from the true distribution.* The bootstrap CI captures **sampling variability** — how much your estimate would change if you ran the exact same experiment again with a different random seed.

It does *not* capture **model misspecification** (if Bradley-Terry's "single skill per agent" assumption is wrong, the CI is honest about sampling noise but not about that).

### What the CIs tell you

| Pattern | Interpretation |
|---|---|
| Tight, non-overlapping CIs | Ranking is statistically reliable. Report differences confidently. |
| Wide, overlapping CIs | You need more games. The point estimate may flip with more data. |
| Asymmetric CIs (one-tailed) | Usually a competitor with very few or very lopsided matches. Add games against varied opponents. |
| CI that includes zero (on log scale) | You can't reject "this agent is exactly average." |

A practical rule we use: if the CIs for ranks N and N+1 overlap by more than ~50%, don't claim a real ordering — they're tied within noise.

### When bootstrap CIs lie

Two failure modes:

- **Too few resamples.** 1000 is a usual default. 100 is not enough — the percentile estimates themselves become noisy. We use 1000 unless the underlying fit is slow.
- **Highly correlated data.** If your matches aren't independent (e.g. all 50 came from one tournament with a fixed bracket), bootstrap underestimates the true uncertainty. We mitigate by sampling scenarios randomly rather than by fixed pairings — see [Chapter 17 §scenarios](../part6-evaluation-arena/17-ranking-pipeline.md).

---

## C.6 What the model assumes (and when it breaks)

Bradley-Terry assumes **transitivity** — if A beats B and B beats C, then on expectation A beats C. Real life sometimes violates this (rock-paper-scissors). For agent prompts the assumption is usually fine: a "more careful" prompt tends to beat "less careful" prompts across most scenarios. But if you have prompts specialized for different scenarios (rush vs. boom vs. defense), pooling all matches into one BT model is misleading — you've collapsed three different rankings into one. The fix is to fit BT *per scenario* and report per-scenario rankings, which is what our [Chapter 17 §"per-scenario fits"](../part6-evaluation-arena/17-ranking-pipeline.md) section does.

---

## C.7 Why not Elo?

Elo is the most famous pairwise rating system. Mathematically it's an *online* (one-update-at-a-time) approximation of Bradley-Terry. We use the offline BT MLE rather than online Elo because:

- **All our data is available at fit time** — there's no streaming requirement.
- **Order-independence.** Elo's rating depends on the order matches are processed; BT MLE doesn't.
- **No K-factor to tune.** Elo's K-factor controls how fast ratings update; choosing it badly inflates or smooths the ratings spuriously.
- **Confidence intervals come naturally from bootstrap.** Elo would need a separate uncertainty estimator (Glicko, TrueSkill).

Elo is great for live chess ladders. BT-MLE is better for the "we have N games, what's the ranking *now*" question.

---

## Further reading

- Hunter, D. R. (2004). *MM algorithms for generalized Bradley-Terry models*. Annals of Statistics 32(1). The canonical reference for the MM solver we use.
- Bradley, R. A., & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika 39. The original paper.
- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993). The reference textbook on the resampling method.
- Cattelan, M. (2012). *Models for Paired Comparison Data: A Review with Emphasis on Dependent Data*. Statistical Science 27(3). Survey including extensions for ties, multiway comparisons, and time-varying skills.

---

**Cross-references:**

- [Chapter 17 — Ranking Pipeline](../part6-evaluation-arena/17-ranking-pipeline.md) — how the MM solver, smoothing, and bootstrap are wired into the arena CLI.
- [ADR 0004 — Bradley-Terry over simple win-rate](../adr/0004-bradley-terry-ranking.md) — the formal decision write-up.
- [Glossary](../glossary.md) — one-liners for MLE, MM algorithm, bootstrap, confidence interval.
