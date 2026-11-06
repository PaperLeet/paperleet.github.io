# Barlow Twins: cross-correlation and redundancy reduction

## 1. Problem

Two augmentations of one sample should preserve its meaning, but constant representations also satisfy a simple agreement objective. [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html) targets a feature cross-correlation matrix: corresponding coordinates should agree while different coordinates carry less redundant information. Implement `cross_correlation(z1, z2, eps=1e-5)` and `barlow_twins_loss(z1, z2, lambd=.0051, eps=1e-5)`.

Both inputs are projected `[B,D]` representations, with corresponding rows paired. Return a `[D,D]` matrix and a scalar loss. There is no negative queue, predictor, or stop-gradient in this objective.

## 2. Core equations

For each view, independently center and standardize each feature across the batch. The [official implementation](https://github.com/facebookresearch/barlowtwins/blob/main/main.py) uses training-mode `BatchNorm1d(affine=False)`. This exercise therefore uses **population variance** and epsilon inside the square root:

$$
\mu_i^{(v)}=\frac1B\sum_b z_{bi}^{(v)},\quad
\sigma_i^{2(v)}=\frac1B\sum_b(z_{bi}^{(v)}-\mu_i^{(v)})^2,\quad
\hat z_{bi}^{(v)}=\frac{z_{bi}^{(v)}-\mu_i^{(v)}}{\sqrt{\sigma_i^{2(v)}+\epsilon}}.
$$

Rows of the correlation matrix index first-view features; columns index second-view features:

$$
C_{ij}=\frac1B\sum_b\hat z_{bi}^{(1)}\hat z_{bj}^{(2)},\qquad
\mathcal L=\sum_i(C_{ii}-1)^2+\lambda\sum_i\sum_{j\ne i}C_{ij}^2.
$$

This follows the objective in Equations 1–2 and the official code's BN stabilization. Positive epsilon makes an identical nonconstant feature's correlation slightly below 1. Both penalties are **sums**. Include both off-diagonal triangles; cross-correlation is generally asymmetric.

Example: if the views coincide and contain centered, mutually orthogonal columns with population variance 1, then `C=I/(1+eps)`. Duplicating one such column creates two equal off-diagonal penalties, not one.

## 3. From equations to code

1. Subtract means over `dim=0`, preserving the `[B,D]` layout.
2. Average squared centered values over `dim=0`, add `eps`, then take square roots and divide.
3. Compute `normalized1.T @ normalized2 / B`. The result compares features, not a `[B,B]` sample-similarity matrix.
4. Sum squared diagonal errors around 1 and all other squared entries around 0; weight only the second sum by `lambd`.

The loss should call the current file's `cross_correlation`. Preserve gradients through both views, their means, and their variances. Keep the supplied `_check_inputs` and `_check_scalar` validation helpers.

## 4. Exercise conventions

- `B>=2, D>=1`. Inputs share shape, device, and float32/float64 dtype and contain finite values. Preserve dtype and device, support noncontiguous inputs, and do not mutate them.
- `eps` is a finite positive Python int/float; `lambd` is finite and nonnegative. Reject bool and Tensor scalar arguments.
- Divide both variance and cross-correlation by `B`. No running statistics, affine parameters, or distributed aggregation are involved in this single-batch core.
- `D=1` has no off-diagonal penalty. `lambd=0` keeps only the diagonal objective. Constant features standardize to zero. Fully constant views have loss `D` and zero gradients at exact collapse.
- Fixed epsilon breaks exact scale invariance. Scaling both inputs by a nonzero `s` and epsilon by `s²` preserves the correlation matrix exactly; `s=0` makes epsilon zero and violates the input contract.

Tests cover an independent scalar covariance oracle, matrix VJPs and loss gradients for both views via finite differences, independent feature translations, paired row/column permutations, view swapping, replicated batches, single features, and tiny variance.

## 5. Interactive charts and CPU experiment

The Lab shows the actual optimization objective, independently computed off-diagonal energy, and all nine matrix entries before and after training. Two directly trainable `16×3` embedding tables contain 96 parameters. Deterministic trigonometric coordinates and distinct mixing matrices initialize correlated features. Adam performs **120 actual updates** with learning rate `.04`, `lambd=.2`, and `eps=1e-5`; record every ten steps.

T3 trains with your loss and independently checks its final matrix and objective using Python scalar arithmetic. Acceptance requires final loss below 1% of its initial value, diagonal error below `.005`, and off-diagonal energy below `.01`. The experiment's `.2` weight suits this small teaching task; the function default `.0051` follows the official setting. This synthetic table experiment does not train an image encoder or reproduce ImageNet results.

## 6. Running

Fill in both functions and select **Run tests**. The numerical core and complete tests support CPU/WASM without a GPU, downloaded dataset, or pretrained model. CLI:

```bash
paperleet test representation/barlow_twins --impl problems/representation/barlow_twins/solution.py
paperleet mutate representation/barlow_twins
paperleet ci --problem representation/barlow_twins
paperleet vizdata representation/barlow_twins
```

## 7. Further reading

Compare [SimCLR](https://proceedings.mlr.press/v119/chen20j.html), which separates sample-level positive and negative pairs, with this feature-level matrix objective. Compare [VICReg](https://arxiv.org/abs/2105.04906), which explicitly separates invariance, variance, and within-view covariance. Its `B-1` sample-variance convention cannot replace this exercise's BN convention.

Read [Equations 1–2 and Algorithm 1](https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf) alongside the [official `BarlowTwins.forward`](https://github.com/facebookresearch/barlowtwins/blob/main/main.py). When `D>B-1`, how does centered representation rank constrain the identity target? Why does penalizing collapse not guarantee escape from an exactly constant initialization?
