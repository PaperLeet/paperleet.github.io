# Online Softmax and Block Attention

[FlashAttention](https://arxiv.org/abs/2205.14135) computes exact attention while reducing transfers between GPU memory levels. Its key numerical ingredient is an online softmax that merges tiles without materializing the entire attention matrix. Read §3.1 and Algorithm 1; the author's production kernels are in [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## Your core module

### Return requirements

#### `online_softmax_update` — Required

Return `(new_max, new_normalizer, new_numerator)` in exactly this order, as a Python tuple of floating tensors shaped `[Q]`, `[Q]`, `[Q,Dv]`. Q is the number of query rows and Dv the value width. These are the updated row maximum, unnormalized exponential mass and unnormalized weighted-value sum over all processed key blocks; the numerator is not yet the normalized attention output. Preserve dtype/device, autograd and all input state tensors; return updated state rather than updating in place.

#### `block_attention` — Provided

Return `(output, logsumexp)`, a Python tuple of floating tensors `[Q,Dv]` and `[Q]`. Q is total query length and Dv value width. output is normalized attention; logsumexp is one natural-log sum-exp of scaled scores over all keys per query, not an attention-weight matrix. Preserve q/k/v dtype/device and their gradient paths, including gradients from logsumexp. Inputs remain unchanged.

### Formula and conventions

Implement `online_softmax_update`. The supplied `block_attention` traverses query/key tiles, scales scores by `sqrt(D)`, and performs final normalization. Return the new row maximum, normalizer and **unnormalized** weighted sum.

For previous state `(m,l,u)` and a new score block `S` with values `V`:

$$
m'=\max(m,\max_j S_j),\quad \alpha=e^{m-m'},\quad P_j=e^{S_j-m'},
$$
$$
l'=\alpha l+\sum_jP_j,\qquad u'=\alpha u+PV.
$$

After all blocks, output `u/l` and log-sum-exp `m + log(l)`. Rescale **both** old mass and old weighted values when the running maximum changes.

| Item | Contract |
|---|---|
| `scores`, `values` | `[Q,K]`, `[K,Dv]`, finite scores and nonempty key block |
| State | `row_max [Q]`, `normalizer [Q]`, `numerator [Q,Dv]` |
| Initial state | `-inf`, zero, zero |
| Wrapper inputs | `q [Nq,D]`, `k [Nk,D]`, `v [Nk,Dv]`; positive integer block size |
| Wrapper output | Attention `[Nq,Dv]` and log-sum-exp `[Nq]` |
| Autograd | Preserve gradients through every block; do not modify input state |

This module deliberately uses finite scores without masks or dropout. Causal/all-masked semantics are covered by the Transformer module.

## Hand-calculated example

Two single-key blocks have scores `0` and `log(3)`, with scalar values `2` and `6`. Their unnormalized masses are 1 and 3, so the final output is `(1*2 + 3*6)/4 = 5` and log-sum-exp is `log(4)`.

## Toy experiment and limits

The supplied 37-token experiment compares block sizes 1, 3, 8, 16, 37 and 64 against independently computed dense attention. It reports maximum output/LSE error and the theoretical size of the largest score tile. Partial edge tiles must work correctly.

**Provenance:** PaperLeet CPU teaching adaptation of Algorithm 1, with synthetic tensors. It is not a paper benchmark or a reproduction of GPU speedups. Autograd may retain intermediate blocks; theoretical score-tile counts do not claim measured training-memory savings. Fused kernels, GPU IO scheduling and recomputed backward are outside the exercise.

## Validation

T0 covers rectangular inputs and partial tiles; T1 uses the analytic two-block example; T2 checks partition invariance, extreme score stability, reusable state, and dense gradient equivalence. T3 runs the submitted tile recurrence over six block sizes and exports a result table. Typical mistakes are failing to rescale one state component, replacing the running maximum with the latest block maximum, or detaching previous blocks.
