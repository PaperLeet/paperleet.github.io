# Decoupled AdamW Update

For adaptive optimizers, adding an L2 penalty to the gradient and directly shrinking parameters are different operations. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), Algorithm 2, separates these operations. The author's experimental code is in [AdamW-and-SGDW](https://github.com/loshchil/AdamW-and-SGDW).

## Your core module

### Return requirements

#### `adamw_step` — Required

Return `(new_parameter, new_first_moment, new_second_moment)` in exactly this order, as a Python tuple of three floating tensors with the inputs' full shape/dtype/device. The moment outputs are the updated uncorrected running moments for the next call, not bias-corrected moments. This is a functional update: return new values and leave parameter, gradient and both input moments unchanged. Retain tensor autograd paths; the caller decides whether to disable gradients. Do not return None as an in-place optimizer step would.

### Formula and conventions

Implement one pure function, `adamw_step`. Given a parameter tensor, its gradient, previous first/second moments and a **one-based** step, return the new parameter and both updated moments. State initialization and the quadratic training loop are provided by the experiment.

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t),\qquad
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

**Exercise convention:** `lr = eta` and `weight_decay = lambda` use the common PyTorch AdamW parameterization. The paper separates a base step size, decay coefficient and schedule multiplier; this exercise folds the learning-rate schedule into `lr`. Decay is multiplied by `lr`, and epsilon is **outside** the square root.

| Item | Contract |
|---|---|
| Tensor arguments | Same shape, dtype and device; second moment nonnegative |
| State | Start both moments at zero; preserve all input tensors |
| Step | Positive integer, starting at 1 |
| Returns | `(new_parameter, new_first_moment, new_second_moment)` |
| Autograd | Preserve tensor gradients in this functional operator; the caller can use `no_grad` |
| Scope | Standard AdamW; no AMSGrad, clipping, maximize option or sparse gradients |

The gradient and moments never receive a weight-decay term. With zero gradient and zero history, the update is simply `parameter * (1 - lr * weight_decay)`.

## Example

Let parameter = 2, gradient = 3, `lr=.1`, `weight_decay=.2`, and zero moments. At step one, corrected moments are 3 and 9. Ignoring tiny epsilon, the new parameter is `2*.98 - .1 = 1.86`. The returned moments still depend on the chosen betas, even though their corrected first-step values do not.

## Toy experiment

The provided 200-step experiment minimizes a quadratic with different curvature in its two coordinates. It records the trajectory and objective values while calling the submitted `adamw_step`. The target is `[.25,-.5]`; the curvature is `[1,16]`, so one coordinate has a much steeper gradient.

**Provenance:** PaperLeet teaching adaptation of the update rule. This is synthetic optimizer behavior, not the original paper's image-classification/generalization experiment. All computations are CPU-only and require no data download. The deterministic final-loss gate is `.01`; the separate reference preview is not a learner result.

## Validation

T0 checks the functional interface and input preservation. T1 derives the first step and probes epsilon placement with a small gradient. T2 checks decay's Jacobian, moment independence from decay, multiple updates against `torch.optim.AdamW`, and coordinate permutation. T3 verifies quadratic optimization and records its trajectory. Mutants exercise coupled decay, missing bias correction, epsilon placement and omitted learning-rate scaling.
