# Common pitfalls

Scores accumulate conditional log-probabilities. Keeping only the last token or comparing raw logits across prefixes is wrong. Different additive shifts at each prefix must leave decoding unchanged.

Expand every retained prefix before global pruning. A per-parent greedy choice loses competing sequences. Exact ties use complete token tuples for deterministic ordering.

Filter unknown full IDs and duplicate items before taking k. Keep original scores without catalog-only normalization. Return catalog row indices, not token values or beam indices.

The original paper allows invalid IDs; distinguish Section 4.5 filtering from its prefix-matching extension. Exhaustive enumeration is practical only for this 625-sequence toy oracle. Finite beam search is approximate and wider beams need not improve arbitrary models monotonically.
