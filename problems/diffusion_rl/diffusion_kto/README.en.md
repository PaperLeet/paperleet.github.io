# Diffusion-KTO Binary Feedback and Utility Baseline

## 1. Problem

Estimate a nonnegative baseline from unrelated pairs and optimize sigmoid utility for binary feedback.

Work with precomputed per-transition log probabilities; implement KTO utility and stop-gradient semantics without an image backbone.

## 2. Core equations

$$
b=\max(0,\operatorname{mean}\operatorname{sg}(\ell_{unrelated})),\quad z=w\beta(\ell_\theta-\operatorname{sg}(\ell_{ref})-b),\quad L=-\operatorname{mean}(\sigma(z)-1/2).
$$

## 3. From equations to code

```python
kto_loss(current: Tensor, reference: Tensor, labels: Tensor, unrelated_log_ratios: Tensor, beta: float) -> Tensor
```

Average unrelated log ratios and clamp the baseline at zero. A desirable label retains the relative-policy direction; an undesirable label reverses it. Center sigmoid utility at zero and negate it for gradient descent.

## 4. Contract

current/reference/labels: equal (B,) tensors; labels are exactly +1 or -1. unrelated_log_ratios: nonempty (M,). beta>0. Return mean negative centered sigmoid utility, with baseline max(mean(unrelated),0).

Only current log probabilities receive gradients. Reference, feedback labels and the unrelated-batch baseline are detached.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A scalar policy preference offset is optimized on balanced desirable/undesirable labels for 60 steps with a separate fixed baseline batch.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/diffusion_kto --impl problems/diffusion_rl/diffusion_kto/solution.py
python -m paperleet.cli mutate diffusion_rl/diffusion_kto
```

## 7. Extensions

Compare saturation under sigmoid, log-sigmoid and linear utility.

[Aligning Diffusion Models by Optimizing Human Utility](https://arxiv.org/abs/2404.04465) · Section 4.1 Eqs. (7)-(8), empirical nonnegative unrelated-pair baseline
