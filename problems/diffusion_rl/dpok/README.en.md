# DPOK Gaussian KL Bound and Policy Gradient

## 1. Problem

Compute per-step Gaussian KL and combine reward score gradients with direct KL gradients.

Implement the efficient gradient approximation in Eq. (9), not the full state-visitation derivative of Eq. (8).

## 2. Core equations

$$
K_{bt}=\frac{\|\mu_{bt}-\mu^{ref}_{bt}\|^2}{2s_t^2},\quad L=\frac1B\sum_b[-\alpha\operatorname{sg}(r_b)\sum_t\ell_{bt}+\beta\sum_t K_{bt}].
$$

## 3. From equations to code

```python
gaussian_kl(current: Tensor, reference: Tensor, std: Tensor) -> Tensor
```
```python
dpok_surrogate(logp: Tensor, rewards: Tensor, kl: Tensor, alpha: float, beta: float) -> Tensor
```

Equal-covariance Gaussian KL reduces to a variance-weighted squared mean distance. Sum over time for the trajectory bound, average over samples, and combine the frozen-reward score term with differentiable KL.

## 4. Contract

gaussian_kl: current/reference (B,T,D), shared std (T,)>0; return (B,T). dpok_surrogate: logp/kl (B,T), terminal rewards (B,), alpha>0, beta>=0; sum time and mean batch.

Current Gaussian means and KL penalties differentiate. Reference means, scales and terminal rewards are frozen; supplied logp differentiates. State-visitation gradients omitted as in Eq. (9).

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A Gaussian scalar policy with variance 1 learns a quadratic target under beta=0.4 KL to a zero-mean reference; 70 updates are compared against the analytic regularized optimum 2/(2+beta).

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/dpok --impl problems/diffusion_rl/dpok/solution.py
python -m paperleet.cli mutate diffusion_rl/dpok
```

## 7. Extensions

Derive the omitted state-visitation gradient term in Appendix A.3.

[DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381) · Section 4.1 Lemma 4.2 and Eqs. (7)-(9), efficient gradient approximation
