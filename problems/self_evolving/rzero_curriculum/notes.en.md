# Implementation notes

Count along each question’s rollout axis. Votes [1,1,0,0] tie, so this exercise chooses 0 with consistency 0.5. Against frozen label 1, fresh answers [2,2,1] earn [0,0,1].

Smallest-ID tie breaking is a deterministic teaching convention. Include both 0.25 and 0.75. Do not refresh the label from fresh answers: that changes the training target. Answer equivalence is provided upstream.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2508.05004
