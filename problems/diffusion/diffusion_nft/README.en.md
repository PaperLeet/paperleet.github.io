# DiffusionNFT Positive and Negative Flow Matching

## 1. Exercise goal

Implement [DiffusionNFT Eq. (5)](https://arxiv.org/html/2509.16117v1#S3.SS2).

## 2. Interfaces and returns

### Return requirements

#### `implicit_velocities` (required)

Return the tuple (positive, negative), in that order. Both are Tensors matching current shape [B,...] and dtype/device; B is batch size and trailing axes are feature dimensions. Keep both branches differentiable through current, detach old inside the function, and leave inputs unchanged.

#### `nft_loss` (required)

Return a scalar Tensor of shape (), with current dtype/device. Sum squared errors over all non-batch coordinates within each example, then average over B examples; do not average over all elements. Gradients flow only into current; old, target and rewards are stopped. Do not mutate inputs.

#### Provided local helpers

- `_check_velocity`: Provided: return Python None when velocities and beta satisfy the contract, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.

### Equations and interface notes

One trainable velocity `v_current` implicitly represents two policies around a frozen rollout velocity `v_old`:

$$
v^+=(1-\beta)v_{old}+\beta v_{current},\qquad
v^-=(1+\beta)v_{old}-\beta v_{current}.
$$

For a forward-process velocity target `v` and an optimality probability `r` in `[0,1]`, optimize

$$
\mathcal L=\operatorname{mean}_{b}\left[r_b\|v_b^+-v_b\|_2^2+(1-r_b)\|v_b^--v_b\|_2^2\right].
$$

Use **beta**, not its reciprocal, in the implicit branches. Negative examples train the negative branch with a nonnegative squared error; they are not a negative-weight MSE.

### Implement

| Function | Inputs | Outputs |
|---|---|---|
| `implicit_velocities(current, old, beta)` | Same-shaped `(B,...)` velocities and positive finite beta | Positive and negative velocities, in that order |
| `nft_loss(current, old, target, rewards, beta)` | Three matching velocities and `(B,)` rewards | Scalar loss |

Sum squared errors over **all non-batch coordinates**, then average over examples. This fixes the convention for image or token-shaped features. Rewards are already normalized probabilities; do not clip, normalize, or broadcast them across different samples. Validation is supplied, with nonempty CPU float32/float64 inputs.

Gradients must flow only into `current`. Detach `old` inside `implicit_velocities`; detach `target` and `rewards` inside `nft_loss`, even if those inputs accidentally require gradients. Preserve input values and dtype/device.

Forward noising, rollout data and training loops are provided scaffolding. The toy uses `x_t=(1-t)*x0+t*noise`, so its velocity target is `noise-x0`. Online rollout generation, reward models and old-policy EMA updates are outside this module.

## Toy: positive, negative and neutral examples

**Provenance: teaching adaptation.** A fixed two-dimensional forward-process minibatch has five rewards: `0`, `0.25`, `0.5`, `0.75`, `1`. Optimize its velocity outputs with your loss and compare with the independently derived quadratic optimum:

$$
v_{current}^*=v_{old}+\frac{2r-1}{\beta}(v-v_{old}).
$$

Positive and negative examples move in opposite directions; a neutral example stays at the old velocity. Charts show loss convergence and the per-reward velocity trajectory. This deliberately small output-space experiment does not claim online policy improvement or image-generation benchmark reproduction.

T0 checks contracts; T1 uses independent scalar formulas and endpoint rewards; T2 checks analytic gradients, frozen context, neutral anchoring, reflection symmetry, feature reduction and closed-form optima. T3 trains with your submitted functions and records the charts. The preview uses the reference implementation.
