# DAPS Decoupled Posterior Sampling

## 1. Exercise goal

Implement the local posterior-sampling kernel in [DAPS, Eq. (7) and Algorithm 1](https://arxiv.org/abs/2407.01521). At a noise level, a supplied probability-flow ODE estimate `x0_hat` defines a Gaussian approximation to the clean sample. Measurement conditioning happens in clean space, followed by independent re-noising to the next level.

## 2. Interfaces and returns

### Return requirements

#### `posterior_score` (required)

Return per-sample conditional scores as a Tensor [B, D], matching x shape and dtype/device; B counts samples and D unknown coordinates. Measurement width M is not an output axis. Do not reduce across samples/coordinates or divide by B or D. Keep differentiable tensor operations and leave inputs unchanged.

#### `langevin_step` (required)

Return the updated sample Tensor with exactly x shape and dtype/device ([B,D] in this exercise, for B samples and D coordinates). Return only the new sample, not the score or noise. eta=0 gives x numerically. Do not mutate inputs or explicitly detach the output.

#### `renoise` (required)

Return one next-level noisy sample Tensor with exactly x0 shape and dtype/device ([B,D] here). sigma_next=0 gives x0 numerically. Keep differentiable tensor operations and do not mutate x0 or the supplied noise.

#### Provided local helpers

- `_positive`: Provided: return Python None when the scalar is valid, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.
- `_matching`: Provided: return Python None when floating tensors have finite values and matching shape/dtype/device, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.

### Equations and interface notes

For a linear observation `y = A x + measurement_noise`, use:

$$
g(x)=\frac{\hat x_0-x}{r^2}+\frac{A^T(y-Ax)}{\beta^2},\quad
x' = x+\eta g(x)+\sqrt{2\eta}\epsilon,\quad
x_{\mathrm{next}}=x_0+\sigma_{\mathrm{next}}\epsilon'.
$$

Here `r` and `beta` are standard deviations. The two noises are independent standard-normal draws supplied by the caller. The final formula centers the next noise level on the MCMC clean sample, with standard deviation `sigma_next`.

### Implement

| Function | Inputs | Output |
|---|---|---|
| `posterior_score(x, x0_hat, operator, measurement, r, beta)` | `x,x0_hat: (B,D)`, `operator: (M,D)`, `measurement: (B,M)` | `(B,D)` conditional score |
| `langevin_step(x, score, eta, noise)` | Matching tensors and a nonnegative step size | One Langevin update |
| `renoise(x0, sigma_next, noise)` | Matching tensors and a nonnegative noise standard deviation | A sample at the next noise level |

Use row-vector batches, so predicted measurements are `x @ operator.T`. Scores are per-sample derivatives: do not divide by the batch size or the number of coordinates. Keep `x0_hat` fixed throughout the inner MCMC loop. Preserve dtype/device and input values; validation is supplied. Tests use CPU float32/float64. No implicit broadcasting across samples is allowed.

The ODE, measurement operator, schedules, random numbers and integration loop are scaffolding. You implement these three mathematical kernels. There are no model downloads or diffusion-network training tasks.

## Toy: an analytic local conditional

**Provenance: teaching adaptation.** Start with an ODE estimate at the origin in two dimensions, a Gaussian conditional approximation with `r=1`, and the observation `x[0]+0.5*x[1]=1.5` with measurement standard deviation `0.5`. Run 4,096 chains for 240 vectorized Langevin steps, then independently re-noise at `sigma_next=0.3`.

The harness computes the conditional mean and covariance analytically. It also computes the finite-step Langevin stationary covariance; a fixed nonzero step size has discretization bias. Charts show measurement consistency, convergence to the conditional mean, and covariance. This experiment validates one DAPS transition, not an entire diffusion inverse-problem solver or the paper's nonlinear Appendix G toy.

T0 checks contracts; T1 differentiates an independent scalar energy and checks explicit-noise updates; T2 checks Gaussian stationary points, batch independence, null-space behavior, noise moments and a composed transition. T3 calls your functions and records the toy artifacts. Preview plots use the reference implementation.
