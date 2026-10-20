# Implementation notes

Compute triangular uncertainty, then the cluster share over the whole batch, then floor at zero and gate by format. With p=0.5 and two duplicates in a batch of four, reward is 0.5; invalid format returns zero.

Agreement is not accuracy: unanimous wrong answers still give p=1. Invalid rows remain in this exercise’s clustering denominator. Observed rewards are detached.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2508.05004
