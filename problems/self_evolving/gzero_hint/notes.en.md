# Implementation notes

Align both scoring passes to the same response tokens; the hint itself is excluded. Token shifts [2,1] yield delta=1.5. Empty rows return zero. Invalid format replaces delta and length terms with -1, while retaining the cluster penalty.

Delta is a probability-shift proxy, not measured quality gain or a full KL divergence. Do not compare different responses. Length is in characters. Keep negative rewards. Hint-delta is differentiable; final rewards are fixed.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2605.09959
