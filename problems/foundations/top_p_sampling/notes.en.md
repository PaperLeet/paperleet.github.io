# Implementation notes

Compare the cumulative mass before each token to p. Keeping only cumulative sums <=p drops the crossing token. Apply temperature before selecting the nucleus and restore vocabulary order afterward.

Common mistakes:

- Drop the threshold-crossing token.
- Keep one extra token at exact threshold equality.
- Fail to restore vocabulary order.
- Ignore temperature before nucleus selection.

## Retain the threshold-crossing token

Compute temperature-scaled full-vocabulary probabilities before sorting. For `[0.5,0.25,0.15,0.1]` at p=0.6, retain the first two entries: the first alone is insufficient. Renormalization gives `[2/3,1/3]`.

Keep tokens whose preceding cumulative mass is <p. Exact equality does not keep another token. Handle p=1 explicitly to retain the entire vocabulary despite rounding, then map the result back to vocabulary order.

Temperature changes full probabilities and the nucleus itself, so apply it before selection. The reference takes O(R·V log V).

## Numerical range and measured sampling

Use float64 intermediates so a positive Python temperature is not rounded to zero in float32. At temperature >=1 divide before subtracting the maximum; at temperature <1 subtract before dividing. Negative overflow then represents a vanishing weight, while the maximum stays zero. Cast the final probabilities back to the input dtype.

T3 samples 8192 token IDs from your returned distribution with a fixed seed, checks that excluded tokens never appear, and plots measured frequencies against independently specified probabilities. The statistical tolerance is for sampled frequencies; deterministic probability checks remain strict. This is a small synthetic experiment, without a trained language model or text-quality benchmark.

## Paper definition and exact boundaries

This exercise follows the shortest-prefix definition in [Holtzman et al., ICLR 2020, §3.1, Eq. (2–3)](https://arxiv.org/abs/1904.09751): cumulative mass >=p. The [authors’ released filtering code](https://github.com/ari-holtzman/degen/blob/0acfd2d0ba8484e24e9c5241f75f34be15ef2609/gen.py#L118-L132) shifts a strict >p mask and retains one additional token at exact equality. Here uniform four-token probabilities with p=.5 retain exactly two. Probability ordering and boundary comparisons use float64; exact ties keep smaller vocabulary indices. This is an explicit teaching convention, not a literal reproduction of the released filter.
