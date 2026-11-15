# ADR 0004: Bradley-Terry ranking for arena profiles

Status: accepted.

## Context

The arena compares a handful of profiles (prompt variant, model, temperature) by racing them in the synthetic world. Each race ends with one final state per profile, and model sampling makes repeated races disagree. The comparison has to produce one number per profile, say how much that number would move with more races, and stay finite when a profile wins every race.

## Decision

Score each final state with a lexicographic tuple: age index, then population, then food plus wood. Turn each race into pairwise wins between the profiles that took part. Fit a Bradley-Terry model to the win matrix with the Minorization-Maximization update, adding 0.5 phantom wins in each direction for every pair. Report mean-centred log-ratings with 95% percentile-bootstrap intervals from a seeded generator. Implementation: `apps/arena/src/ranking.py` (`composite_score`, `_solve_bt`, `_bootstrap_ci`), described in [ranking pipeline](../arena/ranking.md).

## Consequences

- One `aoe2-arena rank` run gives a rating and an interval per profile. Overlapping intervals mean the run cannot separate those profiles.
- An undefeated profile, or a run where every race ties, still yields finite ratings. The prior pulls ratings towards 0, and the pull matters less as races accumulate.
- The same outcomes and `bootstrap_seed` reproduce the same intervals.
- Age dominates the score. A Dark Age profile with a large stockpile never outranks one that reached Feudal.
- Ratings are pooled across scenarios and do not persist between runs. Each run is self-contained.
- The solver raises `RankingError` if it has not converged after 1000 iterations; bootstrap resamples that fail this way are dropped.
- The arena depends on `numpy`.

## Alternatives considered

| Option | Why not |
|---|---|
| Win rate | Gives no interval and ignores who a profile played against |
| Elo | Result depends on race order and on a K-factor; no natural interval |
| TrueSkill or Glicko | More machinery and dependencies than a few profiles need |
| Weighted sum score | Lets a large Dark Age stockpile outrank reaching Feudal |
