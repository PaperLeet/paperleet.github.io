# Rotation Trick Quantization Gradients

[Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/html/2410.06424v1), ICLR 2025 Oral, changes the backward map of vector quantization. The selected code `q` is already provided. Implement `rotation_quantize(e, q, eps=1e-8)` so the forward result equals `q`, while the encoder receives a rotated and rescaled gradient.

## Implementation requirements

### Return requirements

#### `rotation_quantize` (required)

Return one floating `Tensor` with the same `[...,D]` shape, dtype and device as e and q (D>=2); its forward values equal q. Keep the surrogate gradient to e through the frozen scaled rotation, but no gradient to q or the rotation/scale construction. Do not detach the output, return Jacobians or mutate either input.

#### `_validate` (provided)

Provided helper: return `(e_norm, q_norm)`, two floating Tensors of shape `[...,1]`, reducing only the final D features of e/q `[...,D]` and retaining the dimension. Preserve dtype/device and their norm gradient paths; do not mutate inputs. Raise `ValueError` for invalid shape/epsilon, near-zero norms or antiparallel pairs. This returns norms, not `None`.

### Formula and conventions

Let

$$u=e/\lVert e\rVert,\quad v=q/\lVert q\rVert,\quad r=(u+v)/\lVert u+v\rVert,\quad s=\lVert q\rVert/\lVert e\rVert.$$

Section 4.2 gives the efficient transformation

$$R=I-2rr^T+2vu^T,\qquad \widetilde q=\operatorname{sg}(sR)e.$$

Apply it without materializing a `D×D` matrix. Freeze **all directions and the norm ratio**. For upstream column gradient `g`, the encoder gradient must be `s Rᵀ g`, and this surrogate path must not update `q`. The reconstruction-gradient path differs from the separate codebook/commitment objectives used by a complete VQ-VAE.

## Contract

Inputs have identical nonempty `[...,D]` shapes with `D≥2`, finite values, shared CPU float32/float64 dtype/device, and may be noncontiguous. Preserve shape, dtype, device and input values. The supplied validation requires positive finite `eps`, both vector norms greater than `eps`, and `||u+v|| > eps`.

Zero vectors and exactly opposite directions do not define this construction and raise `ValueError`; do not invent a fallback gradient. Positive collinear vectors are valid: `q=3e` gives forward `q` and encoder gradient `3g`.

This is a **surrogate Jacobian**. Finite-differencing nearest-code selection would measure a different function, so tests use independent analytic rotation matrices rather than numerical derivatives of quantization.

## Toy experiment

**Paper-derived adaptation.** Figures 2–3 motivate the quadratic `f(x,y)=x²+y²` and a set of 16 code vectors. We independently choose a 4×4 grid, sample encoder vectors on a circle, and compare your autograd gradients with STE. Charts show gradient angles and magnitudes. The original function/code count are retained; the grid, sampling and line plots are our adaptation, not a reproduction of VQ-VAE training.

The preview uses the reference; T3 calls your implementation and checks the forward values, preserved angle and predicted norm scaling. [Author code](https://github.com/cfifty/rotation_trick).
