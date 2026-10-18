# Kimi K3: Block Attention Residuals

Depth can be an attention axis too. Implement the core selection operator in [Kimi K3 v2, §2.2, Eqs.8–10](https://arxiv.org/html/2607.24653v2#S2.SS2), using the [authors' Attention-Residuals pseudocode](https://github.com/MoonshotAI/Attention-Residuals/blob/master/README.md) as an additional source. The K3 report is fixed to the August 7, 2026 version.

## Your core module

### Return requirements

#### `assemble_sources` — Required

Return one floating Tensor `[N,B,T,D]`, with depth/source count N, batch size B, token count T and feature width D. Source order is embedding, completed_blocks in their original order, then partial_block when supplied. N=1+len(completed_blocks)+(partial_block is not None); a zero-valued supplied block still counts. Preserve dtype/device and gradients to every source without modifying the tensors or caller's list.

#### `depth_attention` — Required

Return `(mixed, weights)` as a Python tuple of floating tensors. `mixed[B,T,D]` aggregates original source values; `weights[N,B,T]` gives each depth source's probability and sums to one over N for every batch/token. N is source count, B batch size, T token count and D feature width. Preserve sources' dtype/device and gradients through sources, query and norm_weight; do not mutate inputs.

### Formula and conventions

`assemble_sources` collects the token embedding, completed block sums, and the current partial block if present. `depth_attention` uses a learned pseudo-query to retrieve from those sources. The surrounding backbone and the calculation of each block sum are supplied externally; do not implement a complete Transformer.

For sources $v_i$, learned query $w$, and learned RMSNorm scale $\gamma$:

$$
k_i=\frac{v_i}{\sqrt{\operatorname{mean}(v_i^2)+\epsilon}}\odot\gamma,
\qquad a_i=\operatorname{softmax}_{i}(w^\top k_i),
\qquad h=\sum_i a_i v_i.
$$

Normalize keys only. Values preserve their original scale. There is **no** Transformer-style division of logits by $\sqrt D$.

| Argument/result | Contract |
|---|---|
| `embedding` | `[B,T,D]`; always the first source |
| `completed_blocks` | List of `[B,T,D]` completed sums; excludes the embedding |
| `partial_block` | Optional `[B,T,D]`; `None` omits it, an explicit zero tensor includes a real source |
| `sources` | Stacked `[N,B,T,D]`, with depth as the leading axis |
| `query`, `norm_weight` | Learned `[D]` vectors shared across tokens for this layer |
| `eps` | Positive scalar added **inside** the RMS square root; default `1e-6` |
| Returned mix, weights | `[B,T,D]`, `[N,B,T]`; weights sum to one over depth, independently per token |
| Numerical scope | Finite, shared floating dtype/device; preserve input lists/tensors, all gradients and strided inputs |

At the first layer of a new block there is no current partial sum. Appending a zero placeholder would change the softmax denominator. Once a real partial sum exists, include it even if its entries happen to be zero. Completed blocks already hold sums of their layer outputs; do not add the embedding into each one again.

## Toy experiment

Three synthetic sources are supplied: an embedding `[1,0]`, a completed block `[-2,0]`, and a current partial block `[0,3]`. Sweep a pseudo-query from `[0,0]` to `[6,0]`. The first chart shows the depth weights; the second shows the distance to the useful embedding. Scalar probabilities calculated independently from these explicit sources serve as the oracle.

**Provenance:** PaperLeet teaching adaptation of the paper's operator and official pseudocode. This is a deterministic query sweep, not model training or an experiment reported by the authors. T3 runs your assembly and attention functions, and the results page displays your actual curves separately from the reference preview.

## Validation

T0 checks source order, shapes and input preservation. T1 uses a token-by-token oracle with PyTorch RMSNorm. T2 checks absent versus zero partial blocks, the uniform-query limit, source/query/scale gradients, depth permutation, noncontiguous sources and stable logits. T3 verifies the synthetic depth-selection behavior.
