# Pitfalls

- Nucleus selection is not the same as a random draw. Always returning argmax loses diversity.
- Sampling sorted ranks without recovering vocabulary IDs returns the wrong tokens.
- Use the caller's generator without reseeding. Shared draws across rows create spurious correlations.
- Inverse-CDF intervals are half-open: equality belongs to the next nonzero interval. Use C_i>u, not C_i>=u.
- Normalize the CDF by its final entry. Retain float64 for accumulation and the uniform draw.
- This suite intentionally specifies one uniform per row. `multinomial` is a valid general categorical sampler but does not satisfy this exercise's exact stream contract.
- T3 measures actual learner-produced tokens; it does not run a pretrained language model.
