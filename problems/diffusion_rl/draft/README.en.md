# DRaFT-LV Reward Gradients from Re-Noised Samples

## 1. Problem

Reuse generated samples via re-noising and one-step denoising to aggregate reward gradients.

Implement DRaFT-LV stop-gradient and averaging for extra samples; the initial differentiable generated sample is supplied.

## 2. Core equations

$$
z_k=\alpha\operatorname{sg}(x_0)+\sigma\operatorname{sg}(\epsilon_k),\quad \hat x_k=(z_k-\sigma\epsilon_\theta(z_k))/\alpha,\quad L=-\frac{r(x_0)+\sum_{k=1}^K r(\hat x_k)}{K+1}\text{ averaged over batch}.
$$

## 3. From equations to code

```python
draft_lv_loss(generated: Tensor, noises: Tensor, alpha: float, sigma: float, denoiser, reward) -> Tensor
```

Retain the original generated sample’s reward gradient. Freeze that sample when constructing additional noisy inputs, reconstruct with the trainable denoiser and evaluate rewards. Average over the batch and then over original plus extra branches.

## 4. Contract

generated (B,D), noises (K,B,D), K>=1; alpha>0 and sigma>=0 amplitudes. denoiser(x) returns (B,D); reward(x) returns (B,). Include original reward and K reconstructed rewards, averaged over K+1 and B.

Original generated samples retain their reward gradient. Re-noising uses generated.detach() and frozen noises; additional gradients reach only the denoiser. Reward parameters are supplied/frozen by caller but input gradients remain live.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A scalar generator and epsilon denoiser share one parameter; two fixed symmetric re-noising draws provide extra differentiable quadratic rewards over 55 updates.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/draft --impl problems/diffusion_rl/draft/solution.py
python -m paperleet.cli mutate diffusion_rl/draft
```

## 7. Extensions

Compare gradient-estimator variance under a fixed sampling budget.

[Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) · Section 4, DRaFT-LV and Algorithm 1 re-noising lines
