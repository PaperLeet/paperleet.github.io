# ZeRO-2 Sharded Adam and State Memory — Solution notes

## Derivation and implementation

Reduce over gradient replicas, not parameter owners. Reshape the averaged gradient to (R,S) to simulate reduce-scatter. Owners update independently and flattening simulates all-gather. Stages 1, 2 and 3 successively shard the 12-, 2- and 2-byte categories.

$$
g_s=\frac1R\sum_r g^{(r)}_{sS:(s+1)S},\quad m_s=\beta_1m_s+(1-\beta_1)g_s,\quad v_s=\beta_2v_s+(1-\beta_2)g_s^2,\quad p_s\leftarrow p_s-\eta\frac{m_s/(1-\beta_1^t)}{\sqrt{v_s/(1-\beta_2^t)}+\epsilon}.
$$

## Boundary and gradient ownership

Parameters and moments have shape (R,S); replicated gradients have shape (R,R*S). Return updated parameter shards and both moments. Average across data-parallel ranks; step starts at one and weight decay is absent. Byte accounting uses two-byte weights, two-byte gradients, and twelve bytes for master weights plus two moments. Parameter count is divisible by R.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Two ranks process deterministic regression examples for 45 Adam steps at learning rate 0.04. Compare with an independent unsharded Adam update and report losses and model-state bytes.

Single-process ZeRO-2 semantics with equal-sized shards and a mixed-precision Adam-state byte model; activations and communication buffers are excluded.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.38814764 |
| final_loss | 0.0031833261 |
| trajectory_error | 4.1633363e-16 |
| replicated_bytes | 16384 |
| stage2_bytes | 3840 |

## Common mistakes

- m1_local_grad: Using only one data rank omits other replicas.
- m2_missing_bias_correction: Omitting first-moment correction changes Adam updates.

[ZeRO](https://arxiv.org/abs/1910.02054) · Sections 5.1–5.3: optimizer, gradient and parameter partitioning.
