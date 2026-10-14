# After passing

Multiply positions by frequencies. Pair i uses `base**(-2*i/D)`; changing the pair index, denominator or feature layout changes the operator.

A reversed rotation still preserves norms, composes and survives joint shifts. Directional hand calculations and an independent scalar oracle are necessary in addition to invariants.

Cached positions belong to the original sequence: a newly appended chunk need not start at zero. Different q/k lengths can use separate position vectors. Do not rotate values.

Keep x and floating positions connected to autograd. The feature VJP is inverse rotation of the upstream vector; position VJPs sum contributions over batch, heads and feature pairs. Casting float32 phases back to float64 cannot recover lost precision.

Adjacent and half-split pairing describe different feature layouts. They require a matching permutation of projected features, not an interchangeable drop-in convention. Joint-shift equality is a mathematical identity; the synthetic cache and logit plots do not establish a trained model's long-context quality.
