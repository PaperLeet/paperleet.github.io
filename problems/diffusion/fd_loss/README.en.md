# FD-loss Differentiable Fréchet Objective

## 1. Exercise goal

Implement the training objective in [Representation Fréchet Loss for Visual Generation](https://arxiv.org/html/2604.28190), using generated population statistics and frozen reference statistics.

## 2. Interfaces and returns

### Return requirements

#### `frechet_distance` (required)

Return one FD scalar Tensor of shape (), preserving input float32/float64 dtype/device. Reduce D-dimensional means and DxD covariances to one distance without dividing by D. Preserve gradients through generated mean/covariance and stop reference_mean/reference_covariance. Gradient guarantees require both covariances to be positive definite; semidefinite inputs have forward-only guarantees. Do not clamp the final FD or mutate inputs.

#### `normalized_fd_loss` (required)

Return a scalar Tensor of shape (), matching distances dtype/device. Sum weighted normalized distances over R representations, without dividing by R or the weight sum. Retain gradients only through numerator distances; detach denominator distances and weights. Leave inputs unchanged.

#### Provided local helpers

- `_check_gaussian`: Provided: return Python None when mean and covariance satisfy the Gaussian-statistics contract, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.
- `_check_pair`: Provided: return Python None when generated and reference statistics have matching shapes/dtype/device, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.
- `_check_normalization`: Provided: return Python None when distances, nonnegative weights and positive denominators are valid, not a boolean or a validated tensor. Invalid inputs raise ValueError. Validation only: leave inputs and their gradient settings unchanged. No returned tensor shape or dtype/device applies.

### Equations and interface notes

For feature Gaussians, Eq. (2) is

$$
FD=\|\mu_g-\mu_r\|_2^2+\operatorname{tr}(\Sigma_g)+\operatorname{tr}(\Sigma_r)
-2\operatorname{tr}\left[(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\right].
$$

The symmetric sandwich has the trace-square-root required by the original covariance-product formula. Appendix B.3 computes its eigenvalues with `torch.linalg.eigvalsh`, avoiding eigenvector differentiation through generated statistics. Covariance matrices generally do **not** commute.

For representation distances `d_i`, Eq. (6) uses

$$
\mathcal L=\sum_i w_i\frac{d_i}{\operatorname{sg}(d_i)+\epsilon}.
$$

### Implement

| Function | Inputs | Output |
|---|---|---|
| `frechet_distance(mean, covariance, reference_mean, reference_covariance)` | `(D,)` means and `(D,D)` covariances | Scalar FD |
| `normalized_fd_loss(distances, weights, eps=.01)` | Matching nonempty `(R,)` vectors and positive finite epsilon | Weighted **sum** of normalized distances |

All tensors share dtype/device, using float32 or float64. Validators are provided. Preserve dtype and input values. Inside the functions, detach reference mean, reference covariance and representation weights. Generated mean/covariance and the distance numerators must keep their gradients.

Use an eigendecomposition of the **detached** reference covariance to build its square root. Form the sandwich, symmetrize it as `(A+A.T)/2`, then sum the square roots of its eigenvalues. Clamp numerical negative eigenvalues to zero before taking their square roots. Do not add diagonal jitter, take an elementwise matrix square root, clip the final FD, or pass a nonsymmetric covariance product to `eigvalsh`.

Forward tests cover positive semidefinite covariances, including zero eigenvalues. **Gradient guarantees require both generated and reference covariances to be positive definite**, including cases with repeated positive eigenvalues. The derivative at a zero eigenvalue is outside this exercise; no hidden regularization changes the objective.

For 1D means `1` and `0` with variances `4` and `1`, FD is `1+(2-1)^2=2`. Normalizing one such distance with weight `1` and epsilon `.01` returns `2/2.01`, but its derivative with respect to that distance is **`1/2.01`**. The denominator is stopped. Tiny negative FD roundoff is accepted as long as `distance+eps` stays positive; distances are not silently clamped.

The official default uses equal weights and epsilon `.01`. This interface also accepts other fixed nonnegative weights and positive epsilons. It does not divide by the number or sum of weights. Multiple frozen feature models, their statistics and image encoders are supplied context, not functions to train here.

## Toy: post-train a 2D generator

**Teaching adaptation.** A five-parameter affine generator transforms four symmetric noise points. Supplied EMA moments (`beta=.8`) feed your FD and normalization functions, which train the generator for 200 steps toward a correlated reference Gaussian. This uses CPU tensors, epsilon `.1` and a small Adam optimizer; it does not reproduce an ImageNet generator or the paper's production hyperparameters.

Charts separately show raw EMA FD, current-generator FD from an independent 2D oracle, and the normalized training objective. The normalized value can stay close to one while raw distance changes substantially, so judge matching progress using **raw FD**, not the near-unit value alone.

T0 checks contracts; T1 uses diagonal and noncommuting 2D closed forms; T2 checks gradients, frozen references, repeated positive spectra, stopped normalization, symmetry and coordinate changes. T3 actually trains with your submitted functions. Preview charts use the reference solution.
