# Review: Causal Recommendation and Target Masks

- Keep the diagonal: the current item is observed. Hide future items, not the current one.
- Mask padding queries and keys. Multiplying the final output by zero does not prove padded keys never affected real queries.
- History IDs and next-item targets are different objects. The objective excludes positions using the target IDs.
- Two binary terms are not a pairwise score-difference objective. A common logit shift usually changes this loss.
- Mask ignored logits before softplus to avoid `0 * NaN`. An all-padding zero must retain both gradient paths.
- Training accuracy measures this small history-dependent synthetic task. Real recommendation needs held-out data, candidate filtering and a specified metric protocol.
