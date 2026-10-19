# Muon Split: Head-wise Orthogonalization

[GLM-5 technical report, v2](https://arxiv.org/abs/2602.15763v2), §2.1, “Multi-latent Attention”. The report splits the attention up-projection matrices $W^{UQ}, W^{UK}, W^{UV}$ by head before matrix orthogonalization. This lets heads receive independently scaled updates.

## Your implementation

### Return requirements

#### `headwise_orthogonalize` — Required

Return one floating Tensor `[num_heads*head_dim,input_dim]` matching matrix's shape/dtype/device. It concatenates the per-head partial polar factors in original consecutive row-head order, not a list or a tensor with an extra head axis. Singular directions below the declared relative cutoff contribute zero; an all-zero head stays zero. Preserve autograd only on full-rank blocks with distinct singular values (rank boundaries/repeated values are excluded from the gradient contract). Do not modify matrix.

### Formula and conventions

Implement one function, `headwise_orthogonalize(matrix, num_heads)`.

The matrix is shaped `[num_heads * head_dim, input_dim]`; each head occupies a **consecutive row block**. Split into `[num_heads, head_dim, input_dim]`, orthogonalize each block, and merge back without changing the layout. A caller would apply this primitive to each projection's gradient/momentum matrix separately.

For each block $G_h=U_h\operatorname{diag}(s_h)V_h^\top$, return

$$
P_h=U_h\operatorname{diag}(\mathbf 1[s_h>\tau_h])V_h^\top,
\qquad \tau_h=\max(d_h,d_{in})\,\epsilon_{dtype}\,\max(s_h).
$$

Use reduced SVD. A zero matrix returns zero. A rank-deficient matrix keeps only its nonzero singular directions: do not manufacture an update in its nullspace. Positive rescaling of a full-rank head leaves its result unchanged. For a wide full-row-rank head, $P_hP_h^\top=I$; for a tall full-column-rank head, $P_h^\top P_h=I$.

Inputs are float32/float64 CPU tensors. Preserve shape and dtype; do not mutate inputs. Reject a nonmatrix, nonfloating tensor, empty dimension, or nonpositive/incompatible head count with `ValueError`. Preserve ordinary tensor autograd on full-rank blocks with distinct singular values. The generic SVD backward can be undefined at repeated singular values; those derivatives and derivatives at rank changes are outside this exercise. Forward values remain defined there.

## Exact teaching boundary

**This exercise uses an exact SVD partial polar factor.** The report describes per-head matrix orthogonalization but does not fully specify its production kernel, polynomial coefficients, iteration count, numerical rank handling, or all optimizer settings. Our SVD and cutoff are explicit teaching conventions. They isolate the splitting idea, rather than reproducing a particular approximate Newton–Schulz kernel. Momentum formation, learning-rate scaling, weight decay, and the model's training loop are outside the required function.

## Toy experiment

`experiment.py` supplies a two-head scale sweep. Both heads have the same row space; the first head's magnitude varies from $10^{-3}$ to $10^3$. Your function generates the per-head output norms. An independent closed-form whole-matrix projection supplies the comparison: a joint projection can suppress the smaller head, while per-head projection preserves its scale.

This is a **PaperLeet teaching adaptation**, not a toy supplied by the GLM authors or a claim about benchmark accuracy. T3 records actual submitted outputs, not a canned result. The Lab preview is generated from the reference implementation; submitted-run artifacts use your implementation.

## Verification

T0 checks the contract; T1 uses a Gram-matrix eigendecomposition oracle; T2 checks orthogonality, scale and basis invariance, null directions, and full-rank gradients. T3 runs the supplied experiment. No checkpoint, accelerator, or network is needed. [Official model repository](https://github.com/zai-org/GLM-5) is the provenance link; this exercise does not claim to copy unpublished optimizer code.
