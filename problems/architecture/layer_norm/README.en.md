# LayerNorm and Add & Norm

Implement the normalization and residual connection used in [Attention Is All You Need, §3.1](https://arxiv.org/abs/1706.03762). The normalization method comes from [Layer Normalization](https://arxiv.org/abs/1607.06450); here it is a module inside the Transformer paper.

## Your core module

### Return requirements

#### `layer_norm` — Required

Return one floating Tensor with the same `[...,D]` shape as x; D is the final feature width and every leading batch/token dimension is retained. The result is the affine-transformed normalized features, not mean/variance statistics. Preserve x's dtype/device and gradients to x, weight and bias; do not modify inputs.

#### `post_norm_residual` — Required

Return one floating Tensor shaped `[...,D]`, matching x and sublayer_output, with all leading dimensions retained and D the feature width. It contains the normalized residual sum. Preserve their shared dtype/device and gradients to both residual branches, weight and bias; do not update either branch in place.

### Formula and conventions

Implement `layer_norm` and `post_norm_residual` using tensor reductions. The sublayer output and experiment are supplied. This exercise uses the original Transformer's **post-LN** ordering and disables dropout:

$$
\mu=\frac{1}{D}\sum_{i=1}^{D}x_i,\qquad
\sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2,
$$
$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,
\qquad y=\operatorname{LN}(x+s).
$$

Here `s` is a supplied sublayer output. Statistics use the **final feature axis only**, separately for every token and batch member. Variance divides by `D`; `torch.var` requires `correction=0` for this convention. Epsilon belongs inside the square root.

| Input/output | Convention |
|---|---|
| `x`, `s` | Same shape `[..., D]`, with `D >= 1` |
| `weight`, `bias` | Learned affine parameters `[D]`; shared across tokens |
| `eps` | Positive scalar; default `1e-5` |
| Output | Same shape, dtype and device as `x`; no input mutation |
| Supported inputs | Finite float32/float64 tensors; affine tensors share dtype/device |
| Autograd | Preserve gradients to `x`, `s`, `weight`, and `bias` |

Use tensor operations rather than `torch.nn.LayerNorm` or `torch.nn.functional.layer_norm`, so you implement the statistics yourself. The test oracle uses the library operator independently.

## Example and boundary cases

For `[1, 3, 5]`, the mean is `3` and the population variance is `8/3`. With unit scale and zero bias, the result is `[-2, 0, 2] / sqrt(8/3 + eps)`. Constant features, including `D=1`, produce exactly the affine bias.

Adding a scalar to every feature of one token leaves its normalized output unchanged. Scaling a token is only approximately invariant with a fixed nonzero epsilon; scaling epsilon by the square of a positive scale factor makes the identity exact in real arithmetic. Changing a different token or batch member must have no influence.

## Toy experiment

Eight six-feature tokens have different offsets and scales. A supplied deterministic sublayer output is added, then your `post_norm_residual` runs. Plots show token means and population standard deviations before and after normalization; a scalar Python oracle supplies the expected result. With unit affine parameters, the resulting variance is `variance / (variance + eps)`, so it need not be exactly one. Perturbing all other tokens demonstrates independence.

**Provenance:** this is a PaperLeet teaching adaptation of §3.1, not an experiment reported in the paper. Dropout is disabled and no model is trained. T3 collects measurements from the submitted code; the reference preview is separate.

## Validation and scope

T0 checks shape, dtype and input preservation. T1 checks an analytical result with nontrivial affine parameters. T2 covers constant/single-feature rows, tiny variance, token independence, shift/scale identities, library-oracle gradients and residual ordering. T3 runs the supplied experiment.

The original paper adds the residual **before** LayerNorm. Pre-LN blocks place normalization before the sublayer and have a different interface. RMSNorm omits mean centering and is a different normalization operator; it is not the original Transformer's LayerNorm. This exercise covers Add & Norm with an already computed sublayer output, not a complete encoder or decoder.
