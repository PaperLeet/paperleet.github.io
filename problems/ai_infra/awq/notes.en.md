# AWQ Activation-Aware Scaling Calibration — Solution notes

## Derivation and implementation

Each exponent starts from the same original weights: multiply by channel scales, quantize groups, then divide by those scales. Select by calibrated linear-output MSE, not weight-space MSE.

$$
a_j=\operatorname{mean}_i|X_{ij}|,\quad s_j=\frac{\max(a_j^\alpha,10^{-4})}{\sqrt{\max_j s^{raw}_j\min_j s^{raw}_j}},\quad \widehat W=Q(Ws)/s,\quad \alpha^*=\arg\min_{\alpha\in G}\operatorname{mean}(X\widehat W^T-XW^T)^2.
$$

## Boundary and gradient ownership

x:(N,D), weight:(O,D); D is divisible by group_size. bits is an integer 2..8, and alphas is nonempty within [0,1]. Groups contain consecutive input channels of each output row. Quantizer step=max((max-min)/(2^bits-1),1e-5), zero=clip(round(-min/step),0,2^bits-1), with ties-to-even rounding. Clamp raw scales to 1e-4, then normalize by sqrt(max*min). Break equal-error ties by supplied alpha order. Return (scales,dequantized_weight,alpha,float MSE); offline calibration is detached.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Use 16 deterministic calibration samples with one channel amplified eightfold and 3×4 weights under three-bit group quantization. Sweep eleven alpha values from zero to one and compare the selected candidate with alpha zero.

Single-linear-layer AWQ grid search and asymmetric group quantization using mean absolute activation statistics; clipping search and TinyChat kernels are excluded.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| best_mse | 0.010094095 |
| baseline_mse | 0.10756917 |
| chosen_alpha | 0.2 |
| reported_error_difference | 0 |

## Common mistakes

- m1_no_inverse_scale: Leaving quantized weights scaled changes the represented layer.
- m2_peak_activation: Peak activations differ from the required mean-absolute statistics.
- m3_weight_mse: Weight error does not represent activation-weighted output error.

[AWQ](https://arxiv.org/abs/2306.00978) · Section 3.2: activation-aware scaling search; official auto_scale.py _search_module_scale.
