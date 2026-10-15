# v-Parameterization and Inverse Rotation

## 1. What the paper solves

A diffusion network can predict clean data, added noise, or velocity. Salimans and Ho study these parameterizations for distillation into fewer sampling steps. Changing the prediction target for the same training sample changes endpoint stability and the weighting of prediction errors.

This module selects the supporting parameterization technique from Appendix D. Implement only `make_v`, `v_to_x0`, and `v_to_eps`; the other conversion helpers are supplied. Completing this module covers these rotation identities. Teacher–student targets and progressive training remain separate parts of the paper.

## 2. Core equations

### Return requirements

#### `eps_to_x0` (provided)

Return one Tensor of clean data recovered from a noise prediction, matching x_t shape [B,...]; B is batch size and trailing axes are sample coordinates. alpha/sigma are already broadcastable by the input convention; do not add coefficient axes. Preserve the documented matching floating dtype/device and gradients. Do not return multiple parameterizations as a tuple or mutate inputs.

#### `x0_to_eps` (provided)

Return one Tensor of noise recovered from a clean-data prediction, matching x_t shape [B,...]; B is batch size and trailing axes are sample coordinates. alpha/sigma are already broadcastable by the input convention; do not add coefficient axes. Preserve the documented matching floating dtype/device and gradients. Do not return multiple parameterizations as a tuple or mutate inputs.

#### `make_v` (required)

Return one Tensor of angular velocity v, matching x0 shape [B,...]; B is batch size and trailing axes are sample coordinates. alpha/sigma are already broadcastable by the input convention; do not add coefficient axes. Preserve the documented matching floating dtype/device and gradients. Do not return multiple parameterizations as a tuple or mutate inputs.

#### `v_to_x0` (required)

Return one Tensor of clean data recovered from v, matching x_t shape [B,...]; B is batch size and trailing axes are sample coordinates. alpha/sigma are already broadcastable by the input convention; do not add coefficient axes. Preserve the documented matching floating dtype/device and gradients. Do not return multiple parameterizations as a tuple or mutate inputs.

#### `v_to_eps` (required)

Return one Tensor of noise recovered from v, matching x_t shape [B,...]; B is batch size and trailing axes are sample coordinates. alpha/sigma are already broadcastable by the input convention; do not add coefficient axes. Preserve the documented matching floating dtype/device and gradients. Do not return multiple parameterizations as a tuple or mutate inputs.

#### `convert` (provided)

Return one Tensor in the dst parameterization, with pred/x_t shape [B,...], preserving the documented matching floating dtype/device and gradients. When src=dst, return pred directly (the same object); otherwise return the converted prediction. Do not return a dictionary of x0/eps/v or mutate inputs.

### Equations and interface notes

Let $x_0,\epsilon,x_t\in\mathbb R^{B\times D}$, with $\alpha=\sqrt{\bar\alpha_t}$ and $\sigma=\sqrt{1-\bar\alpha_t}$ broadcastable to the input shape, and $\alpha^2+\sigma^2=1$. Here alpha is a square-root signal coefficient, unlike the one-step alpha in problem **6**.

$$x_t=\alpha x_0+\sigma\epsilon,\qquad v=\alpha\epsilon-\sigma x_0.$$

$$x_0=\alpha x_t-\sigma v,\qquad\epsilon=\sigma x_t+\alpha v.$$

Set $\alpha=\cos\phi$ and $\sigma=\sin\phi$; differentiation gives $v=dx_t/d\phi$. The inverse is a two-dimensional orthogonal rotation and requires no division. In contrast, recovering data from noise divides by alpha, while recovering noise from data divides by sigma. Each amplifies prediction errors near the endpoint where its denominator approaches zero. For the same erroneous prediction expressed in all three representations:

$$\|\hat v-v\|^2=\frac{\|\hat x_0-x_0\|^2}{\sigma^2}=\frac{\|\hat\epsilon-\epsilon\|^2}{\alpha^2}.$$

The prediction target therefore also implies a loss weighting. See [Progressive Distillation, Appendix D, Equations 26 and 30](https://arxiv.org/html/2202.00512v2#A4).

## 3. From equations to code

Implement three core functions:

| Function | Inputs and output | Key idea |
| --- | --- | --- |
| `make_v` | `x0, eps, alpha, sigma` → v | Compute the rotation’s tangent |
| `v_to_x0` | `x_t, v, alpha, sigma` → x0 | Apply the inverse rotation directly |
| `v_to_eps` | `x_t, v, alpha, sigma` → epsilon | Apply the inverse rotation directly |

Tensors have shape `[B,...]`; coefficients broadcast to that shape. Preserve shape and dtype without aggregating over the batch. Neither inverse requires division, preserving stability at the pure-signal and pure-noise endpoints.

`eps_to_x0`, `x0_to_eps`, and `convert` are provided. They handle the `1e-8` denominator floor, string validation, and identity conversions. `convert` calls your three core functions.

## 4. Conventions

| Item | This exercise | Common scheduler interfaces |
| --- | --- | --- |
| Input coefficients | Already square-rooted alpha and sigma | Often store alpha_cumprod and take square roots internally |
| Output | Return a Tensor directly | diffusers schedulers usually return a step-output object |
| Endpoint division | Clamp denominator to 1e-8 | Depends on the schedule and stability strategy |
| Data dimensions | Any broadcastable shape | Images usually use B×C×H×W |

## 5. Interactive visualizations

<viz id="param_circle" />

Adjust the rotation angle phi to inspect the position and tangent vectors. The rotation direction determines the tangent's sign. This plot shows the coefficient plane; when high-dimensional independent noise and data are approximately orthogonal, it also suggests a sample-space interpretation.

<viz id="error_amplification" />

Move through time to inspect amplification factors $1/\alpha$ and $1/\sigma$, and understand why a parameterization becomes harder near an endpoint.

## 6. Run your implementation

Open module **7** in the browser, complete the three v-parameterization functions, and run the tests.

Tests use the supplied conversion entry point to check six round trips, then independently check finite differences, endpoints, and the error-weight identity. Distillation training is outside this exercise.

## 7. Further exploration

For teacher–student targets and progressive step reduction, continue reading [Progressive Distillation](https://arxiv.org/abs/2202.00512). Flow Matching uses a different velocity, differentiated with respect to linear time; this module differentiates with respect to a rotation angle.
