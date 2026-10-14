# Rotary Position Embedding

## 1. Problem

[RoFormer](https://arxiv.org/abs/2104.09864), Sections 3.2 and 3.4.2, introduces rotations of projected query/key features. The [authors' repository](https://github.com/ZhuiyiTechnology/roformer) illustrates adjacent-coordinate pairing. Implement `apply_rope`; an attention layer can call it independently for q and k.

Try [sinusoidal position encoding](../positional_encoding/README.md) first, then compare adding position vectors with rotating projected features.

## 2. Core formula

For adjacent pair $i$, let $(a,b)=(x_{2i},x_{2i+1})$, $\theta_i=\mathrm{base}^{-2i/D}$ and $\phi=p\theta_i$:

$$a'=a\cos\phi-b\sin\phi,\qquad b'=a\sin\phi+b\cos\phi.$$

Positive angles rotate counterclockwise. Pair norms are unchanged and

$$\langle R(m)q,R(n)k\rangle=\langle q,R(n-m)k\rangle.$$

A joint position shift leaves logits unchanged. Moving only one side generally changes them. Norm preservation alone cannot establish the correct sign or frequencies.

## 3. Formula to code

`x` has shape `[B,H,L,D]`; `positions` is `[L]`. Broadcast D/2 frequencies into phases `[1,1,L,D/2]`. Rotate the even/odd coordinate slices and interleave the results. Preserve the input's shape, dtype and device without modifying either input.

With D=2, position $\pi/2$ maps `[2,3]` to `[-3,2]`; position zero is the identity.

## 4. Contract

- Finite float32/float64 features; positive even D. B or L may be zero; strided inputs work.
- Pair adjacent coordinates, not the first and second halves. Half-split conventions can be valid with a corresponding feature permutation, but are outside this contract.
- Positions may be signed integers or reals, need not be consecutive, and broadcast across B/H. Compute phases in x's dtype. Differentiable real positions are an explicit teaching extension.
- Retain gradients into x and floating positions. The finite Python numeric base must exceed one; bool is invalid. Reject invalid layouts, position lengths or bases with `ValueError`.
- Full-width rotation only: no partial RoPE, context scaling, masking, value rotation or model training.

## 5. Interactive experiment

<viz id="toy_experiment" />

Compare logits before/after a joint shift with an independent scalar formula. The curve can oscillate; it does not imply monotonic similarity decay for every vector pair.

The nine-key cache processes six keys before appending three. New keys retain positions 6,7,8; restarting at zero changes logits. Every submitted-run curve uses your operator. Reference previews are labeled separately. This is a synthetic PaperLeet teaching adaptation, not a paper benchmark or context-extrapolation evaluation.

## 6. Run your implementation

```bash
paperleet test architecture/rope --impl problems/architecture/rope/solution.py
paperleet mutate architecture/rope
paperleet ci --problem architecture/rope
```

T0 checks the interface, T1 a hand calculation, T2 independent values/properties/gradients, and T3 the submitted operator's cache experiment.

## 7. Further work

Connect the rotation after q/k projections and manage their cache positions explicitly. Explain why joint-shift invariance cannot establish rotation direction. Passing this exercise demonstrates the selected operator, not complete RoFormer training or reliable long-context generalization.
