# Pitfalls and validation notes

## Normalization and matrix orientation

This exercise follows training-mode `BatchNorm1d(affine=False)` in the official `main.py`: center over the batch, divide variance by `B`, and add epsilon inside the square root. Default unbiased sample variance and rowwise LayerNorm are different contracts. `C[i,j]` pairs feature i of the first view with feature j of the second. The matrix is generally asymmetric.

Repeating an entire batch leaves `C` unchanged, distinguishing `B` from `B-1`. Independent feature offsets disappear after centering. Swapping views transposes the matrix while preserving the full objective. Positive epsilon means rescaling inputs alone is not an exact invariance; scale epsilon by the square of the same factor too.

## Loss scale and gradients

Sum both penalties. Do not divide by `D` or `D(D-1)`. Off-diagonal entries are ordered feature pairs, so using only the upper triangle loses independent terms. With `lambd=0`, retain the complete diagonal objective.

Both input views, means, and variances remain differentiable. Separate finite-difference checks cover both the scalar objective and a weighted VJP of every matrix entry using independent scalar arithmetic. Matching forward values cannot justify detaching a branch.

Exactly collapsed views yield zero correlation, loss `D`, and zero gradients. The experiment starts from nonconstant features and does not claim recovery from every initialization. For `D>B-1`, centered representation rank also constrains the identity target.

## Maintained faults

1. Omit centering: breaks shift invariance and covariance values.
2. Use `B-1` variance: violates BN normalization and batch-replication invariance.
3. Normalize rows: changes the featurewise objective.
4. Divide the matrix by `D`: accidentally correct only when `B=D`.
5. Average diagonal errors: changes loss scale.
6. Use only the upper triangle: drops independent lower entries.
7. Detach the second view: forward values match, both-view gradient tests fail.
8. Add epsilon outside the square root: violates stabilization, especially near zero variance.
9. Ignore lambda: changes the redundancy weight.

Each mutant is a complete, independently loadable file with a first-line bug description and at most ten added/deleted lines relative to the reference.

## CPU experiment calibration

Sixteen evenly spaced angles generate `cos(t), sin(t), cos(2t)` coordinates. Two fixed mixing matrices create paired `16×3` views; all 96 entries are optimized directly. Initialization uses no random draws (the suite fixture seed is 42). Adam runs 120 updates with learning rate `.04`, `lambd=.2`, and `eps=1e-5`. The baseline is the actual objective at initialization.

On macOS CPU / Python 3.14.2 / PyTorch 2.14.1, the reference starts at approximately `1.00230136` and finishes at `1.21073e-5`. Final diagonal error is `1.16493e-5`; off-diagonal energy is `2.28994e-6`. The independent scalar matrix differs by at most about `1.1e-16`. Fixed acceptance thresholds require a reduction below 1% of the initial objective, diagonal error `<.005`, and off-diagonal energy `<.01`, with independent oracle agreement. These measure this deterministic teaching optimization, not paper pretraining performance.

## Sources

- [ICML 2021 paper and Equations 1–2](https://proceedings.mlr.press/v139/zbontar21a.html)
- [Authors' implementation: main.py](https://github.com/facebookresearch/barlowtwins/blob/main/main.py)

Verified on 2026-11-06. Statements, code, tests, and the synthetic experiment are independently written PaperLeet teaching material based on the paper objective and official conventions; no image-training pipeline is copied.
