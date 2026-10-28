# DDPO Gaussian Transition Densities and Stepwise PPO

## 1. Problem

Compute joint Gaussian denoising-transition log densities and clipped PPO with terminal advantages broadcast over steps.

Implement the DDPO objective for fixed-variance Gaussian transitions and an online toy bandit, without a diffusion backbone.

## 2. Core equations

$$
\ell=-\frac12\sum_d[(a_d-\mu_d)^2/s^2+2\log s+\log(2\pi)],\quad L=-\operatorname{mean}_{b,t}\min(e^{\ell-\ell_{old}}A_b,\operatorname{clip}(e^{\ell-\ell_{old}},1-e,1+e)A_b).
$$

## 3. From equations to code

```python
gaussian_log_prob(mean: Tensor, actions: Tensor, std: Tensor) -> Tensor
```
```python
ddpo_loss(current: Tensor, old: Tensor, advantages: Tensor, clip: float) -> Tensor
```

An action consists of all feature coordinates, so sum log densities over features. Hold sampled actions and the old policy fixed. Multiply each transition ratio by the terminal advantage and choose the conservative clipped branch with minimum.

## 4. Contract

gaussian_log_prob: means/actions (B,T,D), std (T,) strictly positive; return (B,T) summing D. ddpo_loss: current/old (B,T), advantages (B,), 0<clip<1. Average B,T.

Gaussian actions and std are detached; current means differentiate. PPO old log probabilities and advantages are detached. Very large unfavorable log ratios may overflow; finite tested ratios stay within the declared numerical domain |current-old|<=40.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A scalar Gaussian policy with standard deviation 0.6 draws 128 fresh actions per update; 60 PPO gradient updates optimize a quadratic terminal reward toward target 1.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/ddpo --impl problems/diffusion_rl/ddpo/solution.py
python -m paperleet.cli mutate diffusion_rl/ddpo
```

## 7. Extensions

Add per-prompt running reward statistics instead of a single global baseline.

[Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) · Section 4.3 Eq. (6), importance-sampled DDPO; Appendix D.1
