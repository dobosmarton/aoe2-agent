# Bradley-Terry and bootstrap intervals

The [ranking pipeline](../arena/ranking.md) turns pairwise race results into one rating per profile with the Bradley-Terry model, then estimates how much each rating would move with different races by bootstrapping. Both ideas are short.

## The model

Give each competitor `i` a positive strength `π_i`. Bradley-Terry says `i` beats `j` with probability

$$ P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j} $$

Only ratios matter: multiplying every strength by the same constant changes no probability, so the scale has to be fixed somehow. Working with `r_i = log π_i` and centring the `r` values on 0 does that, and turns the formula into a logistic curve of the rating gap:

$$ P(i \text{ beats } j) = \frac{1}{1 + e^{-(r_i - r_j)}} $$

A gap of 0 is a coin flip, a gap of 1 is about 73%, a gap of 2 about 88%. Elo ratings use the same curve on a different scale.

The model has one parameter per competitor, so it assumes strength is transitive: if A usually beats B and B usually beats C, A usually beats C. Profiles that are strong in some scenarios and weak in others break that assumption, and a single pooled rating then averages over the difference.

## Fitting

Let `W_i` be the wins of `i` and `N_ij` the number of decided games between `i` and `j`. The maximum-likelihood strengths have no closed form. Hunter's Minorization-Maximization (MM) algorithm finds them by repeating

$$ \pi_i \leftarrow \frac{W_i}{\sum_{j \ne i} \dfrac{N_{ij}}{\pi_i + \pi_j}} $$

for every `i`, then rescaling. Each step increases the likelihood and there is no step size to tune. `_solve_bt` stops when no strength moves by more than `1e-6`.

The model has no draws. In the arena a tied race adds a game to neither side, so it carries no information.

## Undefeated competitors

If `i` won every game it played, the likelihood keeps growing as `π_i` grows, and the maximum sits at infinity. A competitor that never won runs off to 0 the same way. If nobody won anything, the denominators are 0.

The arena fixes this by adding 0.5 to every off-diagonal cell of the win matrix before fitting: half a phantom win in each direction for every pair. That is equivalent to a weak prior pulling all strengths towards equal. Its effect is one game's worth of evidence per pair, which matters after three races and barely registers after forty.

## Bootstrap intervals

A rating fitted from 20 races would come out differently from another 20 races. The bootstrap estimates that spread from the data alone:

1. Draw a resample of the observed outcomes, the same size as the original, with replacement.
2. Refit the model on the resample.
3. Repeat many times (the arena default is 1000).
4. For each competitor, take the 2.5th and 97.5th percentile of its fitted rating. That range is the 95% interval.

The idea is that the observed outcomes are the best available stand-in for the process that produced them, so resampling them imitates rerunning the experiment. Because each refit is centred on 0, the interval describes a rating relative to the field.

## Reading the intervals

| What you see | What it means |
|---|---|
| Intervals of two profiles do not overlap | The run separates them |
| Intervals overlap | More rounds are needed before claiming an order |
| An interval is wide or lopsided | Few decided races involve that profile, or its results were one-sided |
| All ratings near 0 with narrow intervals | The races were mostly ties |

The intervals cover sampling noise only. They do not cover a wrong model, such as the scenario effect above, and they assume outcomes are independent. In the arena all profiles in a race start from the same state, and rounds repeat the same four scenarios, so outcomes from one scenario are correlated and the intervals are somewhat optimistic.

## References

- R. A. Bradley and M. E. Terry, "Rank analysis of incomplete block designs: I. The method of paired comparisons", Biometrika 39 (1952).
- D. R. Hunter, "MM algorithms for generalized Bradley-Terry models", Annals of Statistics 32 (2004).
- B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap (1993).
