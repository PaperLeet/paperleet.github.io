# Implementation notes

Select using a deterministic tie order, mask other logits to negative infinity, then apply stable softmax. Do not divide probabilities by temperature. For temperature >=1, divide logits before centering to avoid overflowing opposite-sign differences; for temperature <1, center before division. Use float64 intermediates and cast the final result back.

Common mistakes:

- Keep the smallest logits.
- Multiply rather than divide by temperature.
- Keep one extra candidate.
- Retain all ties at the threshold.

## Selection and temperature

A clear implementation uses a stable descending sort and the first k indices. Equal logits retain original vocabulary order, making tie handling reproducible. An inverse permutation gives each original token its rank and hence its keep mask.

Positive temperature preserves logit ranking, so the Top-k set is unchanged by temperature; only probabilities within that set change. k=V recovers full softmax and k=1 yields a deterministic point mass.

The reference takes O(R·V log V), with R leading-dimension rows. The learner implements candidate probabilities. T3 additionally draws 8192 seeded categorical samples from those probabilities and compares frequencies with an independent target; RNG streams may differ between native Torch and CPU/WASM.

## Numerical range and measured sampling

Use float64 intermediates so a positive Python temperature is not rounded to zero in float32. At temperature >=1 divide before subtracting the maximum; at temperature <1 subtract before dividing. Negative overflow then represents a vanishing weight, while the maximum stays zero. Cast the final probabilities back to the input dtype.

T3 samples 8192 token IDs from your returned distribution with a fixed seed, checks that excluded tokens never appear, and plots measured frequencies against independently specified probabilities. The statistical tolerance is for sampled frequencies; deterministic probability checks remain strict. This is a small synthetic experiment, without a trained language model or text-quality benchmark.
