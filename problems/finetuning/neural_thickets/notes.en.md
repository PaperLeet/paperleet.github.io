# Implementation notes

Broadcast a different scalar noise scale over each candidate, always relative to the same base. RandOpt does not turn candidate i into the starting point for candidate i+1. Scales and Gaussian directions are supplied so tests need not depend on hidden RNG state.

Selection maximizes scores. Stable ties preserve original model indices; voting ties choose the smallest class ID. These rules are deterministic exercise contracts, not theoretical claims from the paper. Count only selected experts, and count along the expert axis independently for every sample. Averaging integer class IDs can invent an answer that nobody selected.

The experiment's selection and test coordinates are disjoint. Labels for the test grid are consulted only after selection and voting. Strong synthetic accuracy alone does not imply a pretrained thicket or a benefit over the best single expert. The paper's method retains multiple experts and ensembles predictions; ES instead performs iterative reward-weighted updates of a single center.

Sources: [Section 4, Algorithm 1](https://arxiv.org/abs/2603.12228), [authors' algorithm overview](https://thickets.mit.edu/), [official code](https://github.com/sunrainyg/RandOpt).
