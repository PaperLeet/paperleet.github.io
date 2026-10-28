# FlashAttention-2 Delayed Normalization and Query Blocks — Solution notes

## Derivation and implementation

Query-row blocks independently own their outputs. Each K/V tile updates a stable maximum, denominator and unnormalized numerator. Rescale both old accumulators when the maximum changes, then normalize once at the end.

$$
m\leftarrow\max(m,\max_j S_j),\quad \ell\leftarrow e^{m_{old}-m}\ell+\sum_j e^{S_j-m},\quad U\leftarrow e^{m_{old}-m}U+e^{S-m}V,\quad O=U/\ell,\quad L=m+\log\ell.
$$

## Boundary and gradient ownership

q,k:(N,D), v:(N,V), returning O:(N,V) and L:(N,). Scores include 1/sqrt(D). query_block and key_block are positive integers. Causal visibility is key_index<=query_index; each row sees itself. Normalize U by l only after streaming. Both outputs retain gradients.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Sweep five uneven block sizes on 13 deterministic tokens, four query/key features and three value features, for causal and full attention; compare outputs and LSE with dense execution.

A CPU specification of single-head square attention, including causal masks, tail blocks, numerical stability and autograd; no GPU warp timing.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| maximum_output_error | 8.8817842e-16 |
| maximum_lse_error | 3.5527137e-15 |

## Common mistakes

- m1_missing_numerator_rescale: A new running maximum requires rescaling old numerator mass.
- m2_unscaled_logits: Removing square-root scaling changes both probabilities and LSE.

[FlashAttention-2](https://arxiv.org/abs/2307.08691) · Section 3.1.1, Algorithm 1; Section 3.3, query-row work partitioning.
