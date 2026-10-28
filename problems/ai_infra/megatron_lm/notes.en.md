# Megatron Column- and Row-Parallel MLP — Solution notes

## Derivation and implementation

Each shard sees all input features and owns only a hidden-channel slice. GeLU sits between local matrix products. Sum second-layer contributions without averaging.

$$
Y=\sum_{r=0}^{P-1}\operatorname{GeLU}(XA_r+a_r)B_r+b.
$$

## Boundary and gradient ownership

x:(N,D), w1:(D,H), b1:(H,), w2:(H,O), b2:(O,). H is divisible by parts. Partition first-layer columns and corresponding second-layer rows. Add b2 once after reduction. Preserve gradients, dtype and input values.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Use fixed 9×3 inputs and eight hidden channels; train output weights with two shards for 30 steps and compare loss, weights and gradients against dense execution.

A single-process tensor-parallel two-layer MLP including biases and gradients, with tanh-approximate GeLU and no distributed communication.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.0072715013 |
| final_loss | 0.00012874584 |
| trajectory_error | 5.5511151e-17 |

## Common mistakes

- m1_average_reduce: Averaging partial outputs changes the linear map.
- m2_bias_per_rank: Adding a replicated bias once per rank multiplies it.

[Megatron-LM](https://arxiv.org/abs/1909.08053) · Section 3, Figure 3: column-partitioned first GEMM and row-partitioned second GEMM.
