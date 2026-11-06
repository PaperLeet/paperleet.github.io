# SimSiam: stop-gradient and symmetric prediction

## 1. Problem

SimSiam matches the prediction from one view to the projection of another. Implement `negative_cosine_similarity(prediction, target)` and `simsiam_loss(p1, p2, z1, z2)` to learn its objective without negatives or a momentum encoder, and the stopped-target derivative that defines its optimization. Inputs are already network outputs; no encoder or predictor implementation is required.

Source: [the paper](https://arxiv.org/abs/2011.10566), Section 3, Equations (1)–(4) and Algorithm 1.

## 2. Core equations

Normalize each row of a `[B,D]` tensor using this exercise's explicit zero-vector convention:

$$
\hat v_i=\frac{v_i}{\max(\lVert v_i\rVert_2,10^{-12})}.
$$

The directional and symmetric objectives are:

$$
D(p,z)=-\frac1B\sum_{i=1}^B \hat p_i^\top\widehat{\operatorname{stopgrad}(z_i)},
\qquad L=\frac{D(p_1,z_2)+D(p_2,z_1)}2.
$$

When both vector norms are at least `1e-12` so the norm floor does not change them, aligned directions give `-1` and opposite directions `+1`; orthogonal vectors give `0`. Negative losses are correct. Sum feature products, average examples, and average the two directional means.

## 3. From equations to code

1. Stop `target` inside the directional function and retain the prediction graph.
2. L2-normalize both inputs along `dim=1`.
3. Average the negative row dot products.
4. Cross the views: `p1↔z2` and `p2↔z1`.

Stopping a target does not freeze the shared encoder. When `p1=h(z1)`, `z1` still receives gradients through the prediction path; its appearance as the other direction's target is stopped. Do not detach predictions, the final scalar, or the whole shared encoder.

## 4. Conventions

All inputs share a finite nonempty `[B,D]` shape, with `B>=1, D>=1`, matching float32/float64 dtype and device. Return a scalar preserving dtype/device, support strided inputs, and leave inputs unchanged. Provided validation rejects empty axes, broadcast shapes, mixed dtypes and NaN/Inf.

The exact clamp is `max(norm,1e-12)`, including zero and near-zero vectors. A zero prediction against a nonzero target has zero loss but generally a nonzero prediction gradient. This epsilon is an exercise convention; the paper primarily considers nonzero representations.

Tests cover an independent Python scalar oracle, prediction finite differences, stopped targets, shared gradient paths, view swaps, paired permutations, positive row rescaling and zero-vector boundaries. No queue, EMA or image network is required.

## 5. Interactive plots and CPU experiment

Two independent `4×3` prediction tables learn two fixed crossed target tables using 80 actual SGD updates with learning rate `2.0`, recorded every four steps. There are 24 prediction parameters. Targets are also offered to the optimizer: only a correct stopped-target derivative keeps them fixed. Losses come from the submitted function; cosine diagnostics come from a separate scalar calculation.

Acceptance requires final mean cosine above `.98`, improvement above `.5`, exactly unmoved targets, and scalar-oracle loss error below `1e-11`. Inputs are fixed asymmetric vectors; framework seed 42, CPU float64. This experiment isolates gradient routing. It does not reproduce shared-encoder image pretraining or establish a universal no-collapse guarantee.

## 6. Run

```bash
paperleet test representation/simsiam --impl problems/representation/simsiam/solution.py
paperleet mutate representation/simsiam
paperleet vizdata representation/simsiam
```

Complete the two functions in the web editor and run the tests. CPU/WASM executes the same complete suite and experiment without a GPU or data downloads.

## 7. Further reading

Compare InfoNCE/SimCLR negative classification, MoCo's momentum targets and dictionary, and VICReg's explicit variance and covariance constraints. Removing stop-gradient leaves forward values unchanged while changing derivatives, so matching loss values alone cannot validate SimSiam.

Sources: [Algorithm 1](https://arxiv.org/html/2011.10566v1), [official target outputs](https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py), and [official symmetric training loss](https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py). Statements, tests, scalar oracle and teaching experiment are independently authored by PaperLeet.
