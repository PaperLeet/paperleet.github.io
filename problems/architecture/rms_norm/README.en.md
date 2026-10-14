# RMSNorm and Partial Feature Statistics

## 1. Problem

Implement `rms_norm` and `partial_rms_norm`. Explore what changes when normalization keeps the mean, and when its scale estimate sees only part of each token. No Transformer model is needed.

The primary source is [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), Equation (4) and Section 5, by Biao Zhang and Rico Sennrich, NeurIPS 2019.

## 2. Core formula

For a vector with $D$ features, this exercise uses:

$$
s_D(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_D(x)}w_i.
$$

The partial variant replaces only the statistic; it still returns every feature:

$$
s_k(x)=\sqrt{\frac1k\sum_{j=1}^k x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_k(x)}w_i,\quad i=1,\ldots,D.
$$

Do not subtract the mean or add a learned bias. Apply `weight` after normalization; it must not change the denominator.

## 3. Formula to code

Reduce squared values over the last dimension and keep that dimension for broadcasting. For pRMSNorm, the denominator uses `x[..., :k]` while the numerator uses all of `x`. Keep the statistic differentiable: a loss on suffix outputs can still affect prefix inputs through the shared denominator.

A hand calculation: `[-3,4,12]` has full mean square $169/3$, while its first two features have mean square $25/2$. Setting `k=D` recovers full normalization.

## 4. Exercise contract

| Item | Contract |
| --- | --- |
| Inputs | `x [..., D]`, `weight [D]`, $D\ge1$, matching floating dtype/device |
| Precision | Finite float32/float64; retain shape, precision, device and gradients; no input mutation |
| Epsilon | Positive and inside the square root: `sqrt(mean_square + eps)` |
| Partial statistic | Explicit integer `k` in $[1,D]$; invalid values and bool raise `ValueError` |
| Scope | Implement both functions with tensor operations rather than a normalization API |

The paper's main formula omits numerical stabilization. The authors' 2019 PyTorch code adds epsilon outside RMS; this exercise chooses epsilon inside the square root. They differ for small inputs. Passing `k` explicitly avoids the percentage-rounding differences between the paper and code.

## 5. Interactive view and experiment

The supplied experiment compares two eight-feature rows: a balanced prefix and a low-energy prefix with a larger suffix. Increase `k` and inspect deviation from full normalization. Error need not decrease monotonically for arbitrary inputs; equality is required at `k=D`.

A second chart adds offsets and plots output mean and RMS. With positive epsilon, scaling inputs alone does not give exact scale invariance. Scaling inputs by $a>0$ and epsilon by $a^2$ does preserve outputs exactly.

**Provenance:** a training-free PaperLeet teaching adaptation. It does not reproduce translation benchmarks or hardware speedups. Tests and submitted-run charts call your functions; reference previews are labeled separately.

## 6. Run

Complete the web editor and run the full suite, or use:

```bash
paperleet test architecture/rms_norm --impl problems/architecture/rms_norm/solution.py
paperleet mutate architecture/rms_norm
```

## 7. Further study

T0 covers interfaces, precision, strided inputs and empty batches; T1 uses hand calculations; T2 checks zero inputs, token independence, scale identities, a library oracle, analytical gradients and finite differences; T3 evaluates the actual submitted prefix-bias and offset experiment.

Compare with the LayerNorm module: the normalized part of a constant vector becomes zero there but remains nonzero here. Derive the partial Jacobian and identify which inputs can influence other outputs through the scale statistic.
