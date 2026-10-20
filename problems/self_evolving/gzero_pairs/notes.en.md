# Implementation notes

Compute the linearly interpolated median over the full pool before applying quality checks, preserving input order. For deltas [-3,-2,10,11], if the first two fail quality, retain nothing; do not refill from the upper half.

Include median ties, so more than half may survive. Negative deltas are allowed. Do not leak hints into prompts or reverse chosen/rejected. Structural quality evaluation is supplied upstream.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2605.09959
