# Common pitfalls

Each distinct book quantizes the remaining residual. Reconstruction sums all selected vectors; residual updates subtract them. Frozen encoding must not build gradients; this is not straight-through training.

Exact ties use the smallest index. Reordering rows preserves each item's quantization but can change collision suffixes. Complete tuples have separate counters starting at zero; a global row number is wrong. Empty batches still return two-dimensional outputs.

Independent scalar distances and reconstruction sums check the algorithm; scaling and row permutation check invariants. Nonincreasing residual error requires zero in every book. Synthetic frozen latents are not RQ-VAE training or a recommendation benchmark.
