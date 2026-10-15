# LightGCN hints

1. Convert histories to floating values and fill both user-to-item and item-to-user blocks. Leave the other blocks zero.
2. Compute row degrees. Multiply by inverse square-root degrees on both endpoints. Row-only division defines a different operator.
3. Keep a current state and an accumulated sum initialized with E0. Perform exactly K graph multiplications, then divide by K+1.
4. Do not detach intermediate states or apply ReLU. The experiment already supplies BPR loss and regularization.
5. An isolated node at depth two must retain one third of its initial vector.
