# FD-loss Population Moments and EMA

## 1. Exercise goal

Implement the population estimator in [Representation Fréchet Loss for Visual Generation, Section 3.2](https://arxiv.org/html/2604.28190). FD-loss estimates a feature distribution using more history than the current gradient batch. The default EMA variant stores a mean and a **raw second moment**, with gradients flowing through current features only.

## 2. Interfaces and returns

### Return requirements

#### `batch_moments` (required)

Return the tuple (mean, raw_second), both Tensors, in that order. For features [B,D], B counts samples and D features: mean is [D] and raw_second is [D,D], the uncentered second moment, not covariance. Both moments use divisor B, not B-1. Preserve features float32/float64 dtype/device and gradients; do not mutate inputs.

#### `ema_population_stats` (required)

Return the tuple (mean, raw_second, covariance), in that order, with Tensor shapes [D], [D,D], [D,D], where D is feature width. Preserve features float32/float64 dtype/device and current-batch gradients (zero when beta=1); detach both historical tensors internally. Do not update history in place; the caller saves detached mean/raw_second separately.

#### Provided local helpers

- `_check_features`: Provided: return Python None when features are a valid finite floating [B,D] tensor, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.
- `_check_history`: Provided: return Python None when current features, historical moments and beta satisfy the contract, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.

### Equations and interface notes

For current features $X\in\mathbb R^{B\times D}$, compute

$$
\mu_b=\frac1B\sum_i x_i,\qquad M_b=\frac1B\sum_i x_i x_i^\top.
$$

Then apply Eqs. (4)-(5):

$$
\mu=\beta\,\operatorname{sg}(\mu_{old})+(1-\beta)\mu_b,\quad
M=\beta\,\operatorname{sg}(M_{old})+(1-\beta)M_b,\quad
\Sigma=M-\mu\mu^\top.
$$

### Implement

| Function | Output |
|---|---|
| `batch_moments(features)` | `(mean, raw_second)`, shapes `(D,)`, `(D,D)` |
| `ema_population_stats(features, old_mean, old_second, beta)` | `(mean, raw_second, covariance)`, in that order |

Inputs use the same float32/float64 dtype and device. `B>=1`, `D>=1`, and `beta` lies in `[0,1]`; endpoint values are included for testing. History contains valid finite population moments. Supplied validators cover shapes, dtypes and finite values. Preserve input values and do not update history in place.

Use division by **B**, including when `B=1`. `raw_second` is `E[xx^T]`, not a centered covariance. Detach both historical tensors inside `ema_population_stats`, even when they require gradients. Keep all returned statistics differentiable through the current batch. The caller saves `mean.detach()` and `raw_second.detach()` as the next history; the training loop supplies this state management.

For example, one current feature `[3]`, historical mean `[1]` and raw second moment `[[2]]`, with `beta=.5`, produce mean `[2]`, raw second `[[5.5]]`, and covariance `[[1.5]]`. Simply averaging the old and current covariances would incorrectly give `[[.5]]`.

This exercise implements the paper's **EMA population convention**. The official FIFO queue branch estimates a sample covariance using `N-1`; that distinct estimator is outside this module. No Bessel correction belongs in the EMA equations.

## Toy: population statistics from tiny batches

**Teaching adaptation.** Two shifted 2D components contain eight feature points in total. Forty microbatches alternate between the components, so each batch poorly represents the whole population. Supplied scaffolding warm-starts EMA from those eight points, then calls your functions to compare batch-only and EMA estimation errors. Charts show the estimated mean, variance, and combined moment error. This controlled stream is not an original paper figure or a guarantee about changing real-world distributions.

The paper warm-starts its estimator from a base generator before post-training. This toy supplies features directly to isolate the estimator. Fréchet loss, image encoders, multi-device gathering and generator optimization are outside this exercise.

T0 checks contracts; T1 uses scalar raw moments and an explicitly weighted population; T2 verifies stopped history, analytic current gradients, decay endpoints, affine transformations and multi-step geometric weights. T3 runs your submitted functions over the feature stream. Preview charts use the reference implementation.
