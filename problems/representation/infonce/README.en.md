# InfoNCE: identify the positive candidate

## 1. Problem

Implement `info_nce_loss(logits, positive_indices)` to raise the probability of the designated positive among `K` candidates for each context. Inputs are already **logits**, and the positive column may differ between rows.

[Contrastive Predictive Coding (CPC)](https://arxiv.org/abs/1807.03748) predicts future representations from context and constructs a learning signal from positive and negative candidates. This exercise implements the InfoNCE objective in Section 2.3, Equation (4). Positive indices express known sample correspondences in the prediction task rather than human category labels.

## 2. Core equations

Let `s[b,j]` be a logit and `p[b]` the positive column. Each row contains one positive and `K−1` negatives:

$$
\ell_b=-\log\frac{\exp(s_{b,p_b})}{\sum_{j=0}^{K-1}\exp(s_{b,j})},\qquad
L=\frac1B\sum_{b=0}^{B-1}\ell_b.
$$

Keep the positive in both numerator and denominator. Compute the stable equivalent:

$$
\ell_b=\operatorname{logsumexp}_j(s_{b,j})-s_{b,p_b},\qquad
\frac{\partial L}{\partial s_{b,j}}=\frac{\operatorname{softmax}(s_b)_j-\mathbf1[j=p_b]}B.
$$

If exponentiated scores are `[2,3,5]` and the positive column is `1`, loss is `log(10/3)`. Equal candidate scores give `log(K)`; one candidate gives zero loss and zero gradient.

## 3. Equations to code

- Reduce `logsumexp` across candidates (`dim=1`) to obtain `[B]`.
- Gather each row's positive logit using `positive_indices`, also obtaining `[B]`.
- Subtract and average over all contexts, returning a scalar Tensor.

Avoid exponentiating large raw logits. Adding a constant to every candidate in one row must leave the loss unchanged. Different contexts have independent softmax denominators. These inputs already specify scores: apply no additional L2 normalization, temperature scaling or stop-gradient.

## 4. Exercise conventions

`logits` is a finite float32/float64 `[B,K]` tensor with `B>=1,K>=1`. `positive_indices` is int64 `[B]` on the same device, with `0<=p<K`. Preserve the floating dtype, device and derivatives of every logit. Accept strided tensors and leave both inputs intact. Supplied `_check_inputs` rejects empty axes, shape/dtype mismatches, invalid indices and NaN/Inf.

Gradients must include each candidate's denominator contribution. Tests cover an independent scalar oracle, closed forms, analytical derivatives, finite differences, context/candidate permutations, row offsets, large scores and actual optimization.

## 5. Interactive plots and CPU lab

The exercise page shows actual optimization of six contexts with four candidates each: a `6×4` score table containing 24 trainable parameters. Initial scores use fixed trigonometric expressions; positive columns are `[1,3,0,2,1,0]`. Run 80 full-batch SGD updates at learning rate `1.5`, logging loss, mean positive probability and strict top-1 accuracy every ten updates. Ties count as incorrect.

This original teaching lab demonstrates how InfoNCE changes candidate probabilities. It contains no CPC encoder or autoregressive network; plotted metrics describe training on the score table. T3 calls your loss and requires all six positives to rank first, mean positive probability above `.9`, loss below `.1`, and agreement with an independent scalar objective.

## 6. Running the exercise

Click **Run tests** on the website for the full CPU/WASM suite. From the repository root:

```bash
paperleet test representation/infonce --impl problems/representation/infonce/solution.py
paperleet mutate representation/infonce
paperleet vizdata representation/infonce
```

Replace `--impl` with your completed starter path when practicing. No GPU, dataset download or pretrained weights are needed.

## 7. Extensions and sources

Continue with [SimCLR](#/representation/simclr), which builds `2B` candidates from two views and excludes each anchor itself before positive classification. [MoCo](#/representation/moco) supplies candidates through a momentum encoder and a queue. These methods specify where scores and candidates come from; this exercise establishes the objective first.

Source: [CPC](https://arxiv.org/html/1807.03748v2), Section 2.3, Equations (4) and (5), Aaron van den Oord, Yazhe Li and Oriol Vinyals, 2018. The paper uses positive scores `f`; this exercise accepts `s=log(f)`. Its mutual-information bound assumes the paper's positive/negative sampling scheme and is not an unconditional estimate for an arbitrary score table. Statements, validation, tests, implementations and synthetic experiment are original PaperLeet teaching material.
