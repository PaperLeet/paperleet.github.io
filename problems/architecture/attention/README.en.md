# Scaled Dot-Product Attention

The Transformer routes information from value vectors according to query/key compatibility. Implement the operator in [Attention Is All You Need, §3.2.1, Eq. (1)](https://arxiv.org/abs/1706.03762). The author's original implementation is part of [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

## Your core module

### Return requirements

#### `scaled_dot_product_attention` — Required

Return the Python tuple `(output, weights)`, both floating tensors: `output[B,H,Q,Dv]` contains value aggregates and `weights[B,H,Q,K]` contains attention probabilities. B is batch size, H head count, Q/K query/key lengths, and Dv value width. Each nonempty allowed-key row sums to one; fully masked rows have exactly zero weights and output. Preserve the inputs' dtype/device and gradients through q/k/v; masked rows have finite zero gradients. Do not modify inputs.

#### `causal_mask` — Provided

Return one `torch.bool` Tensor `[Q,K]`, where Q=query_length and K=key_length; rows are queries and columns are keys. An entry is True exactly when the key index is at most the query index (upper-left alignment). Use the requested device. This visibility mask has no gradients.

### Formula and conventions

Implement `scaled_dot_product_attention`. The `causal_mask` helper and synthetic experiment are supplied. Inputs already contain projected, split heads; projection layers, positional encoding, feed-forward blocks, and complete language-model training are outside this module.

$$
S=QK^\top/\sqrt{d_k},\qquad A=\operatorname{softmax}_{\mathrm{keys}}(S+M),\qquad O=AV.
$$

The mask contribution is zero for visible positions and minus infinity elsewhere. Compute stable softmax along the **key** axis, separately for each query and head. Return both `O` and `A` so the experiment can show where information came from.

| Input/output | Convention |
|---|---|
| `q`, `k`, `v` | `[B,H,Q,D]`, `[B,H,K,D]`, `[B,H,K,Dv]` |
| `mask` | Optional boolean tensor broadcastable to `[B,H,Q,K]`; `True` allows attention |
| Output, weights | `[B,H,Q,Dv]`, `[B,H,Q,K]`; preserve dtype and device |
| Fully masked query | Exact zero output and weights, finite zero gradients |
| Causal alignment | Upper-left: key position `j` is allowed when `j <= i` |
| Scope | Finite inputs, no dropout; `D` and `Dv` need not match |

An all-masked row needs an explicit convention: applying softmax directly to a row of minus infinity gives NaNs. Set that row's logits to a finite placeholder before softmax, then zero the forbidden weights. This padding behavior is a PaperLeet extension to the paper's formula.

## Example

With query `[2,0,0,0]` and keys `[1,0,0,0]`, `[0,0,0,0]`, the scaled scores are `[1,0]`. Their weights are approximately `[0.7311,0.2689]`. A mask `[True,False]` instead produces weights `[1,0]`.

## Toy experiment

The supplied **prefix averaging** experiment sets all query/key vectors to zero. A causal mask then makes each query average exactly the values seen so far. Changing future values by 1000 must leave earlier outputs unchanged. Results include the attention matrix, actual/expected prefix means, and future leakage.

**Provenance:** PaperLeet teaching adaptation of the paper's attention and masking equations, not an original paper experiment. It uses six scalar values and no training or external data. T3 calls the submitted implementation; the reference lab preview is labeled separately.

## Validation

T0 checks shapes and input preservation. T1 derives a two-key result analytically. T2 checks broadcast masks, empty rows, permutation and shift invariance, independent gradients, and causal isolation. T3 runs the prefix-average experiment. Typical errors are forgetting the square-root scale, ignoring the mask, and normalizing over queries.

Implement the required function in the editor and run all four test layers. Completing this module covers the paper's attention operator, not the complete Transformer.
