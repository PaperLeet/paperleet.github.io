# ImageReward Ranking and ReFL One-Step Feedback

## 1. Problem

Implement pairwise reward ranking and one-step clean prediction from a frozen rollout prefix.

Train a small reward scorer and update an epsilon predictor through a fixed differentiable reward; no BLIP or image decoder.

## 2. Core equations

$$
L_{rank}=\operatorname{mean}\,\operatorname{softplus}(r_l-r_w),\qquad \hat x_0=(\operatorname{sg}(x_t)-\sigma_t\epsilon_\theta)/\alpha_t.
$$

## 3. From equations to code

```python
ranking_loss(preferred: Tensor, rejected: Tensor) -> Tensor
```
```python
refl_clean(latent: Tensor, epsilon: Tensor, alpha: float, sigma: float) -> Tensor
```

Form a rejected-minus-preferred score margin and apply stable softplus. Detach the already sampled latent before solving the noising equation for the clean sample. Freezing reward parameters must preserve the derivative with respect to its input.

## 4. Contract

ranking_loss accepts equal (B,) preferred/rejected scores. refl_clean accepts equal (B,D) latent/epsilon, alpha and sigma are amplitude coefficients (not cumulative variances), alpha>0, sigma>=0.

Ranking differentiates both scores. ReFL stops the rollout latent and differentiates epsilon; the frozen reward model must still propagate gradients to its input.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

Eight scalar preference pairs train a linear reward scorer; 30 subsequent reward-gradient steps update a one-step epsilon model. This tests feedback mechanics, not image quality.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/imagereward --impl problems/diffusion_rl/imagereward/solution.py
python -m paperleet.cli mutate diffusion_rl/imagereward
```

## 7. Extensions

Extend the exercise to a differentiable decoder while keeping reward parameters fixed.

[ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) · Section 2.2 Eq. (1); Section 3 Algorithm 1
