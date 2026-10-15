# Differential Transformer Attention

Implement the core of [Differential Transformer §2.1](https://arxiv.org/html/2410.05258v1#S2.SS1), ICLR 2025 Oral. Two attention maps are normalized separately, then subtracted:

$$A_1=\operatorname{softmax}(Q_1K_1^T/\sqrt d+M),\quad A_2=\operatorname{softmax}(Q_2K_2^T/\sqrt d+M),\quad O=(A_1-\lambda A_2)V.$$

`M` is zero on allowed positions and `−∞` on future positions for causal attention. Apply softmax over **keys**, and mask **both** branches.

Also implement the scalar parameterization

$$\lambda=\exp(\lambda_{q1}^T\lambda_{k1})-\exp(\lambda_{q2}^T\lambda_{k2})+\lambda_{init}.$$

## Interfaces

### Return requirements

#### `differential_lambda` — Required

Return one zero-dimensional floating Tensor (`shape == ()`) containing the scalar coefficient lambda, not a Python float or one-element vector. It is not restricted to [0,1]. Preserve the parameter vectors' dtype/device and gradients to all four vectors, without changing inputs.

#### `differential_attention` — Required

Return one floating Tensor `[B,H,T,2D]`, with batch size B, head count H, sequence length T and Q/K width D; its shape matches value. This is the differential value aggregate before head normalization/output projection, not a tuple containing attention weights. Preserve input dtype/device and gradients through both Q/K branches, value and a Tensor coefficient. Do not modify inputs.

### Formula and conventions

- `differential_lambda(q1, k1, q2, k2, lambda_init=0.2)` takes equal nonempty parameter vectors `[Dλ]` and returns a scalar tensor. These are lambda parameters, separate from the attention Q/K tensors. The caller supplies the layer-dependent initialization; the paper uses `0.8 - 0.6 exp(-0.3(l-1))` for layers starting at 1.
- `differential_attention(q1, k1, q2, k2, value, coefficient, causal=True)` takes Q/K tensors `[B,H,T,D]` and V `[B,H,T,2D]`. Return `[B,H,T,2D]`. The coefficient is a finite Python scalar or scalar tensor; preserve its gradient.

Inputs are finite, nonempty CPU float32/float64 tensors, possibly strided. Preserve dtype/device, gradients and values. Numerical tests keep the lambda exponentials finite. Shape and scalar validation is supplied. No dropout is used. Per-head normalization, its fixed multiplier, projections and the full model are outside this exercise.

The signed map has row sum `1−λ`; it can have negative entries and is not a probability distribution. At `λ=0` it becomes ordinary attention. Identical maps cancel at `λ=1`.

## Toy experiment

**Teaching adaptation.** A four-token sequence combines a distinguishable signal with a value offset shared by every token. Sweep lambda from 0 to 1 using your attention and measure the offset's response: exactly `(1−λ)` times that offset. A second curve shows the signal that differing maps retain. This controlled cancellation property does not imply that arbitrary noise always decreases.

The preview uses the reference; T3 uses your functions. Independent scalar softmax and analytic vector-Jacobian products verify outputs and all Q/K/V/lambda gradients. [Author implementation](https://aka.ms/Diff-Transformer).
