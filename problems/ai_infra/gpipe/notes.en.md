# GPipe Fill-Drain Scheduling and Microbatch Gradients — Solution notes

## Derivation and implementation

Forward work follows the preceding stage and microbatch; backward work drains from the last stage and microbatch. Accumulate sums of squared error, then divide by the sample count, rather than averaging chunk means.

$$
F_{s,m}=s+m,\quad B_{s,m}=K+M-1+(K-1-s)+(M-1-m),\quad L=\frac{1}{N}\sum_c\sum_{i\in c}(x_i^T w-y_i)^2.
$$

## Boundary and gradient ownership

stages, microbatches and chunk_size are positive integers. Events are (phase, stage, microbatch, start), with phase F/B and unit duration. All forward events finish before reverse-order backward events. x is (N,D), y (N,), w (D,); a short final chunk is allowed. Divide total squared error by N.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Train a four-feature linear model on 11 deterministic samples for 35 SGD steps at learning rate 0.08; compare full-batch and size-four chunks. Also compute useful-slot fractions for four stages as microbatch count varies.

A CPU model of equal-duration stages and synchronous updates, with linear regression; communication, BatchNorm and rematerialization are outside this kernel.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.97372915 |
| final_loss | 0.0019951675 |
| trajectory_error | 5.5511151e-17 |
| utilization_first | 0.25 |
| utilization_last | 0.84210526 |

## Common mistakes

- m1_chunk_mean: Normalizing by chunks changes the sample-mean gradient.
- m2_overlap_backward: Starting backward before the fill phase ends breaks the synchronous barrier.

[GPipe](https://arxiv.org/abs/1811.06965) · Section 2.2, Algorithm; Figure 2, synchronous microbatch pipeline.
