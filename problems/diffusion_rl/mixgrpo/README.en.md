# MixGRPO Sliding Stochastic Windows and Local Policy Objectives

## 1. Problem

Implement a training-dependent SDE window, ODE transitions outside it and a policy objective averaged only over selected steps.

Implement the basic mixed ODE-SDE formulation, excluding the later CPS extension and high-order solvers.

## 2. Core equations

$$
l=\min(\lfloor m/\tau\rfloor s,T-w),\quad W=\{l,\ldots,l+w-1\}.\quad L=-\operatorname{mean}_{b,t\in W}\min(e^{\ell_{bt}}A_b,\operatorname{clip}(e^{\ell_{bt}},1-c,1+c)A_b).
$$

## 3. From equations to code

```python
window_start(update: int, total: int, width: int, interval: int, stride: int) -> int
```
```python
mixed_transition(state: Tensor, velocity: Tensor, time: float, next_time: float, noise_level: float, stochastic: bool) -> tuple[Tensor,Tensor]
```
```python
window_loss(log_ratios: Tensor, advantages: Tensor, start: int, width: int, clip: float) -> Tensor
```

Advance the window by completed shift intervals and clamp it at the final valid position. ODE steps use velocity integration; SDE steps add score correction and a nonzero scale. Slice the optimized window before averaging the clipped objective.

## 4. Contract

window_start uses zero-based update index, total>=width>=1, stride>=1, interval>=1; saturates at total-width. mixed_transition: (B,D) state/velocity, 0<=next_time<time<1, noise_level>=0; returns mean and scalar std. window_loss: (B,T) log ratios, (B,) advantages, valid start/width, |selected ratios|<=40.

Rollout states are detached, velocity remains differentiable. Advantages are frozen; loss gradients are zero outside the selected window.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A four-transition scalar flow uses a moving two-step stochastic window. Fresh Gaussian paths and terminal quadratic rewards update one velocity parameter for 60 iterations.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/mixgrpo --impl problems/diffusion_rl/mixgrpo/solution.py
python -m paperleet.cli mutate diffusion_rl/mixgrpo
```

## 7. Extensions

Replace the ODE segment with a higher-order solver and compare model evaluations.

[MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802) · Section 3.1 mixed ODE-SDE; Algorithm 1; Section 3.2 Eq. (11)
