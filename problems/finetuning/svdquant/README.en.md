# SVDQuant Smoothing, Low-rank and Residual Quantization

Implement the numerical core of [SVDQuant v1 §4.2, Eq. (5)](https://arxiv.org/html/2411.05007v1). Move activation outliers into the weights, retain dominant weight components at high precision, and quantize only the residual branch.

## Functions

### Return requirements

#### `smooth_inputs` — Required

Return `(xs, ws)`, a Python tuple of floating tensors `[B,M]` and `[M,N]` in the same order as x and weight. B is batch size, M input channels and N output channels; the smoothed pair satisfies xs@ws=x@weight. Preserve shared dtype/device and leave inputs unchanged. This exercise requires inference values, not gradients through the quantization pipeline.

#### `split_weight` — Required

Return `(left, right, residual)` as floating tensors `[M,r]`, `[r,N]`, `[M,N]`, where weight is `[M,N]` and r=rank. left@right is a best rank-r approximation and residual=weight-left@right. Equivalent factor signs/bases are accepted; do not return singular values in place of residual. rank=0 gives empty factors and the full weight as residual; full rank gives a numerically zero residual. Preserve dtype/device without modifying weight; SVD gradients are outside scope.

#### `svdquant_linear` — Required

Return one floating Tensor `[B,N]`, where x is `[B,M]`, weight is `[M,N]`, B is batch size and M/N are input/output channels. This is the combined high-precision low-rank and fake-quantized residual output, not a tuple of branches. Preserve shared dtype/device and input values; full rank recovers x@weight up to numerical error. This is an inference contract with no gradient requirement through SVD/rounding.

#### `fake_quantize` — Provided

Return one floating Tensor with x's shape/dtype/device: quantized-and-dequantized values from the per-tensor symmetric INT grid, not integer codes or a (codes,scale) tuple. An all-zero input returns zeros. Inputs are unchanged; gradients through this provided rounding helper are not part of the exercise.

#### `_check_smoothing` — Provided helper

Return None after validating nonempty x[B,M], weight[M,N] and positive finite scale[M] with matching input-channel sizes. Checked invalid layouts/scales raise ValueError. No tensors are changed or returned.

#### `_check_rank` — Provided helper

Return None when weight is a matrix and rank is a Python int in [0,min(weight.shape)]. Invalid rank/layout raises ValueError; no factorization is returned and inputs are unchanged.

### Formula and conventions

Use `x[B,M]`, `weight[M,N]`, positive finite `scale[M]`; the unquantized output is `x @ weight` (not PyTorch Linear's transposed-weight layout).

1. `smooth_inputs(x, weight, scale)` returns `(xs, ws)` with `xs=x/scale`, `ws=scale[:,None]*weight`. This preserves `xs@ws=x@weight`.
2. `split_weight(weight, rank)` returns `(left[M,r], right[r,N], residual[M,N])`. `left@right` is a best rank-r approximation from truncated SVD and `residual=weight-left@right`. Any singular-vector signs or equivalent factorization are accepted. Support `rank=0` and `rank=min(M,N)`.
3. `svdquant_linear(x, weight, scale, rank, bits=4)` first smooths, then splits the **smoothed** weight, and computes

$$Y=(\hat X L_1)L_2+Q(\hat X)Q(R).$$

The low-rank branch uses **unquantized smoothed activations**. Quantize residual R, not the full smoothed weight. The provided `fake_quantize` uses one scale per tensor: `s=max(abs(z))/(2^(bits-1)-1)`, `Q(z)=s*clamp(round(z/s),-qmax,qmax)`, round ties to even, and zero maps to zero. `2<=bits<=8`.

The starter provides validation and quantization. Inputs are finite CPU float32/float64 with matching dtype/device and nonempty dimensions. Preserve dtype/device, support strided inputs and do not modify inputs. This is an inference numerical contract; gradients through SVD/rounding are outside the exercise.

## Toy experiment

T3 creates synthetic activations with one 30x channel, then recomputes output errors and singular-value curves through your functions. Compare naive W4A4, smoothing only and the two-branch approximation at ranks 0–6. This is a **teaching adaptation** of Figures 4–5, using given smoothing scales and a one-shot SVD. High precision means the input float dtype here, rather than hardware FP16. No checkpoint, calibration search, iterative refinement, FP4 encoding, integer kernels or Nunchaku latency reproduction is included. This is more than an SVD task, but still a bounded numerical core of the complete system.
