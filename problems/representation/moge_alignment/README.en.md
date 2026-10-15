# MoGe Robust Optimal Alignment

Implement the robust optimal alignment (ROE) core of [MoGe](https://arxiv.org/abs/2410.19115). A predicted point map has an unknown shared scale and a shift along the camera's z axis. The alignment objective uses inverse target depth and coordinate-wise absolute residuals:

## Implementation requirements

### Return requirements

#### `optimal_scale_1d` (required)

Return the Python tuple `(scale, loss)` in that order; both are detached scalar floating `Tensor`s with shape `()` and x's dtype/device. `loss` is the sum of weighted absolute residuals, individually capped when `trunc` is set, not a mean. Ties choose the smallest active scale; with no active coefficient return scale zero and its objective. Do not mutate inputs.

#### `roe_align` (required)

Return the Python tuple `(scale, z_shift)`, two detached scalar floating `Tensor`s of shape `()` with the point-map dtype/device. `z_shift` is a scalar translation only along z, not a 3-vector; the second return is not an objective value. Fit only masked-valid points; no input mutation or solver gradient is required.

#### `_check_vectors` (provided)

Provided validator: return Python `None` when x, y and nonnegative weight are finite, nonempty, matching 1D floating tensors and trunc is None or finite positive. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

#### `_check_points` (provided)

Provided helper: return `(valid_prediction, valid_target)`, floating Tensors `[M,3]` in the original mask-selected order, where M is the number of True mask entries. Preserve point dtype/device and leave originals unchanged. Validate nonempty selection and positive valid target depths, raising `ValueError` on checked invalid inputs; masked-out NaNs are excluded before value validation. The helper does not itself detach; the solver callers are no_grad.

### Formula and conventions

$$
\min_{s,t}\sum_{i\in M}\sum_{d\in\{x,y,z\}}
\min\left(\frac{|s\hat p_{id}+t\mathbf 1_{d=z}-p_{id}|}{p_{iz}},\tau\right).
$$

With `trunc=None`, omit the cap. The cap applies **after weighting** each coordinate. Rotation and x/y translation are fixed. Scale is an unconstrained real scalar for this optimization exercise.

## Implement

| Function | Inputs | Outputs |
|---|---|---|
| `optimal_scale_1d(x, y, weight, trunc=None)` | Same-shaped nonempty 1D floating tensors; nonnegative weights | Scalar `scale`, scalar minimum objective |
| `roe_align(prediction, target, mask, trunc=None)` | `(N,3)` point maps and Boolean `(N,)` valid mask | Scalar `scale`, scalar `z_shift` |

The first function solves `sum(min(weight * abs(scale*x-y), trunc))`. A global minimum can be found among active zero-residual breakpoints `y[i]/x[i]`, where `x[i] != 0` and `weight[i] > 0`. Truncation introduces additional kinks, but their slope changes cannot create a strict local minimum. For equal minimum values, choose the smallest active breakpoint; return scale zero if no coefficient is active.

For point maps, enumerate a valid anchor `k` and impose `t = target[k,2] - s*prediction[k,2]`. Subtract this anchor's depth from the z coordinates only; x/y are unchanged. Each anchor reduces the full problem to the 1D solver. Select the candidate with the lowest objective. Any global minimizer is accepted for the full alignment.

The supplied validation requires positive finite valid target depths and matching CPU float32/float64 tensors. Select valid points before arithmetic: masked coordinates may contain NaNs. Do not mutate inputs. Return detached scalar tensors of the input dtype/device. The official training implementation reconstructs differentiable parameters from selected anchors; that gradient path is outside this exercise.

Small exact enumeration is sufficient (`N <= 64` in the harness). Network prediction, multi-scale neighborhoods, and camera recovery are outside this module.

## Toy: recover a corrupted point cloud

**Provenance: teaching adaptation.** Six generated points have a known scale/z-shift and one high-leverage corrupted coordinate. Your functions produce a scale-objective curve and recovered depths. Truncated ROE recovers the inliers while untruncated fitting is distracted by the outlier. This is a new educational experiment, not a native MoGe benchmark.

T0 checks contracts; T1 uses independent scalar and 2D line-arrangement oracles; T2 checks weighting, truncation order, masking, ties and scene-unit equivariance. T3 calls the submitted functions and emits the toy charts. The preview uses the reference implementation.

The algorithm follows [the authors' `align` and `align_points_scale_z_shift`](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/alignment.py); the educational implementation uses direct candidate evaluation instead of production sorting/scatter kernels.
