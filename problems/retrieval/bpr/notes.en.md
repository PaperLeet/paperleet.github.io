# BPR-MF: preserving the intended preference direction

1. **Unobserved is not a negative label.** The mask defines eligible comparisons. Exclude this user's entire observed history, while preserving items seen only by other users.
2. **Direction and axes matter.** Minimize `-log sigmoid(user·positive - user·unobserved)`. Sum features and average triples. Reversing the sign learns the wrong order; averaging features introduces a dimension-dependent temperature.
3. **Keep both factor tables differentiable.** For `a = sigmoid(-margin) / B`, the data gradients of one triple are `-a*(positive-negative)` for the user, `-a*user` for the positive and `a*user` for the negative. Contributions to repeated rows must add.
4. **The prior covers complete tables.** Every parameter receives `2*l2*parameter`. Penalizing gathered occurrences overcounts repeated IDs and omits unselected rows. This exercise's global-prior coefficient is distinct from role-specific sampled-row updates.
5. **Empty batches and stable values.** Empty data still retains a differentiable zero and the prior. Skip table norms when `l2=0` so unused large rows cannot contaminate the result through overflow. Stable softplus/log-sigmoid handles extreme negative margins without `log(0)` saturation.

The synthetic item-only bound counts opposing directions. Exceeding it demonstrates user conditioning on the training task. It does not measure generalization or reproduce the paper's sampler, regularization schedule or dataset evaluation.
