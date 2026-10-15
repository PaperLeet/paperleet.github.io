# Multi-Head Attention

One head routes information using one projected feature space. Multiple heads can retrieve different information from the same tokens. Implement the projection and assembly of [Attention Is All You Need, §3.2.2 and Figure 2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2). The authors' original code is part of [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

## Your core module

### Return requirements

#### `multi_head_attention` — Required

Return `(output, weights)` in that order: floating tensors `[B,Q,D]` and `[B,H,Q,K]`, where B is batch size, Q/K query/key lengths, D model width and H=num_heads. output has already concatenated heads and applied w_o; weights retain each head rather than averaging heads. Preserve input dtype/device and gradients to query/key/value and all four projection matrices. Fully masked query rows are zero; do not mutate inputs.

#### `attention_kernel` — Provided

Return `(output, weights)` as floating tensors `[B,H,Q,Dv]` and `[B,H,Q,K]`: B batch size, H heads, Q/K query/key lengths and Dv value width. Weights sum to one over visible keys; a fully masked row has zero output and weights. Preserve the input dtype/device and q/k/v autograd with finite gradients for masked rows. This helper returns per-head values before concatenation/output projection and does not mutate inputs.

### Formula and conventions

Implement `multi_head_attention`. The stable scaled attention function `attention_kernel` is supplied in the editor, so this module can be completed independently of the scaled attention exercise. Focus on input projections, splitting heads, passing the mask, concatenating head outputs, and the final projection.

$$
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),\qquad
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O.
$$

| Input/output | Convention |
|---|---|
| `query`, `key`, `value` | `[B,Q,D]`, `[B,K,D]`, `[B,K,D]`; query and key lengths may differ |
| `w_q`, `w_k`, `w_v`, `w_o` | `[D,D]`; use `x @ w`, the paper's orientation, not `nn.Linear.weight` orientation |
| `num_heads` | Positive integer dividing `D`; each head has width `D/H` |
| Head assignment | Consecutive projected feature slices belong to consecutive heads |
| `mask` | Optional boolean mask broadcastable to `[B,H,Q,K]`; `True` means visible |
| Returned output, weights | `[B,Q,D]`, `[B,H,Q,K]`; weights remain separate for each head |
| Fully masked row | Zero weights and output, including finite zero gradients for that row |
| Scope | Finite tensors with a shared floating dtype/device; no biases or dropout; preserve autograd and inputs |

Keep the axes explicit: project `[B,Q,D]`, reshape to `[B,Q,H,D/H]`, then transpose to `[B,H,Q,D/H]`. After attention, transpose the head axis back before flattening the last two dimensions. A reshape alone changes which features belong to each query. Cross-attention uses queries from one sequence and keys/values from another.

The fully masked convention is a PaperLeet extension for padding robustness. The supplied kernel handles it. No residual connection or normalization belongs inside this function; those are separate Transformer components.

## Toy experiment

Two queries read three tokens through two heads. The first query's first head prefers key 0; its second head prefers key 2. The supplied experiment computes exact probabilities from the constructed logits, then checks concatenation and a nontrivial output projection. Its charts show each head's key weights and the resulting feature vector against the analytic target.

**Provenance:** PaperLeet teaching adaptation of the paper's multi-head operator. The small synthetic example is not an experiment reported in the paper, and no training or dataset download is required. T3 executes the submitted function; the reference preview is labeled separately.

## Validation

T0 checks the cross-attention interface and input preservation. T1 uses an independent per-head projection oracle with PyTorch's attention operator. T2 checks broadcast and head-specific masks, fully masked rows, all seven input/parameter gradients, noncontiguous inputs, the one-head limit, and causal isolation. T3 checks the two-head routing experiment. Typical mistakes are reusing the query projection for keys, dropping the mask, merging the wrong axes, and forgetting the output projection.
