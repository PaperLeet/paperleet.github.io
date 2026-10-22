# 104. ES at Scale: Reward-Weighted Parameter Updates

## Problem

Implement `reward_zscores` and `es_step`. The caller evaluates perturbed models and supplies their rewards. Your update should favor parameter directions associated with higher rewards. [ES at Scale](https://arxiv.org/abs/2509.24372) applies this approach to full-parameter LLM post-training; this exercise isolates the numerical update.

## Core equations

Section 3, Algorithm 1 uses candidates $\theta+\sigma\epsilon_i$ and reward standardization:

$$
\bar r=\frac1N\sum_i r_i,\quad
s=\sqrt{\frac1N\sum_i(r_i-\bar r)^2},\quad
z_i=\frac{r_i-\bar r}{\max(s,\varepsilon)}.
$$

$$
\theta_{new}=\theta+\frac{\alpha}{N}\sum_i z_i\epsilon_i.
$$

The population-standard-deviation convention and positive floor make singleton/constant populations well-defined. As in the paper's update, `alpha` absorbs the usual inverse-noise-scale factor. Do not divide by sigma again.

## From equations to code

`rewards` has shape `[N]`; `theta` has parameter shape `S`; `noise` has shape `[N,*S]`. Broadcast one reward weight over each candidate's parameter coordinates, then reduce only the population axis. `noise` contains the standard-normal directions, before multiplication by sigma.

For scalar `theta=2`, `noise=[-1,1]`, `rewards=[0,2]`, and `alpha=0.1`, normalized rewards are `[-1,1]` and the new parameter is `2.1`.

## Exercise conventions

- `N>=1`; `S` may be scalar or a nonempty tensor block. Inputs are finite CPU float32/float64 with matching dtype/device. Noncontiguous views are valid; intermediate arithmetic is finite.
- `eps>0`, `alpha>=0`. Clamp the standard deviation itself; do not add eps to variance. Use population variance, not sample variance or reward ranks.
- Both functions return detached tensors in the input dtype/device. This optimizer does not build a differentiable training graph. Preserve all inputs.
- Constant rewards give zero standardized rewards and no update. `alpha=0` leaves weights unchanged. Noise generation and reward evaluation belong to the caller.

## Interactive figures and experiment

The Lab runs your `es_step` for 100 iterations on reward `-||theta-[1,-1]||²`, starting at `[-2,3]`. Each iteration draws 30 candidates with sigma `0.1`; alpha is `0.05`. CPU float64, seed 42, and two threads give a reproducible search curve. The reference reduces squared distance from 25 to approximately 0.01254; acceptance is below 0.08. No backpropagation occurs. This is a synthetic optimizer check, not an LLM result or a comparison of training costs.

## Running the exercise

Complete both functions and use **Run tests**. T0–T2 cover an independent scalar oracle, population replication, reward shifts/scaling, sign reversal, degenerate populations and graph isolation. T3 calls your update throughout the search.

```bash
paperleet test optimization/es_at_scale --impl problems/optimization/es_at_scale/solution.py
paperleet mutate optimization/es_at_scale
```

## Further questions

Why must duplicating every population member leave the update unchanged? Why can 30 sampled directions yield a useful update without reconstructing a billion-dimensional gradient? Compare this iterative update with Neural Thickets' selection and prediction ensemble.
