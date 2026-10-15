# SiTU-GLU Smooth Bounded Activation

Implement the activation core from [Kimi K3 v2 §2.3.2, Eq. (12)](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS2):

$$
c_\beta(z)=\beta\tanh(z/\beta),\qquad
\operatorname{SiTU}(g,u)=c_{\beta_1}(g)\,\sigma(g)\,c_{\beta_2}(u).
$$

The defaults are `beta1=4`, `beta2=25`. The sigmoid receives the **raw gate**.

## Interface

### Return requirements

#### `softcap` — Required

Return one floating Tensor with exactly x's shape, dtype and device, including scalar or empty tensors. Each value is smoothly capped to the range [-beta,beta] (endpoints may be reached numerically). Preserve gradients into x; do not mutate x or return the cap parameter alongside the result.

#### `situ_glu` — Required

Return one floating Tensor matching the shared shape/dtype/device of gate and up, retaining arbitrary leading dimensions and empty axes. It is the elementwise gated activation, bounded in magnitude by beta1*beta2; no feature dimension is reduced. Keep gradients to both branches and leave inputs unchanged.

### Formula and conventions

Implement `softcap(x, beta)` and `situ_glu(gate, up, beta1=4, beta2=25)`. Both branches are already projected, equally shaped floating tensors. Preserve shape, dtype, device, gradients and input values. Support strided and empty tensors. Tests use CPU float32/float64. Positive finite cap validation is supplied in the starter; projections and MoE routing are outside this exercise.

Near zero, softcap approaches the identity, so this activation approximates SwiGLU locally. Each output coordinate is bounded by `beta1 * beta2 = 100`; see Appendix B. Hard clamping changes derivatives and is not equivalent.

## Toy: recompute Figure 4

Set both branches to `x` and compare GLU, SwiGLU and your SiTU-GLU on the published `[-10,100]` domain. Four charts show full responses, both branches, and a near-origin view.

**Paper-derived recreation:** 111 full-range and 81 near-origin samples are independently recomputed, with added numerical checks. No original graphic is copied; this is not model training. The preview uses the reference, while T3 uses your submission.

Checks cover scalar oracles, bounds, analytic gradients, local/limiting behavior and elementwise independence.
