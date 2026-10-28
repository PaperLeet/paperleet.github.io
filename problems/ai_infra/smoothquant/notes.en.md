# SmoothQuant Equivalent Activation and Weight Smoothing — Solution notes

## Derivation and implementation

Reduce activation maxima over samples and weight maxima over output channels to obtain the same input-channel vector. Divide activations and multiply weights by that vector. Calibration statistics are detached while the equivalent matrix product preserves training gradients.

$$
s_j=\max\left(\frac{(\max_i|X_{ij}|)^\alpha}{\max(\max_o|W_{jo}|,\epsilon)^{1-\alpha}},\epsilon\right),\quad \widehat X=X/s,\quad \widehat W=sW,\quad \widehat X\widehat W=XW.
$$

## Boundary and gradient ownership

x:(N,D), weight:(D,O), alpha in [0,1], eps>0. Clamp weight-channel absolute maxima to eps, then clamp final scales to eps. Zero activation channels follow this formula. Calibration scales are detached; smooth_transform is differentiable in x, weight and scale. Inputs are unchanged.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Use 17 samples with one activation channel amplified by 100 and correspondingly small weights; sweep alpha and compare direct and smoothed INT8 reconstruction errors.

Calibration maxima, offline scales and differentiable equivalent transforms; the experiment simulates symmetric INT8 without integer GEMM or LayerNorm fusion.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| unsmoothed_mse | 0.052506032 |
| balanced_mse | 2.6201785e-05 |
| equivalence_error | 3.3306691e-16 |

## Common mistakes

- m1_weight_wrong_direction: Dividing both operands destroys the equivalent linear transform.
- m2_mean_instead_of_max: Mean statistics do not enforce the specified peak-activation scale.

[SmoothQuant](https://arxiv.org/abs/2211.10438) · Section 4, Equations (3) and (4); official smooth.py offline scale construction.
