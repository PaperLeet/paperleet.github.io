# DanceGRPO Shared Initial Noise and Group Rewards

## 1. Problem

Construct shared initial noise per prompt, per-reward group advantages and a transition-level GRPO objective.

Test multiple rewards and shared initial conditions in a small Gaussian policy; no video backbone or visual reward model.

## 2. Core equations

$$
A_{pg}=\sum_k\frac{R_{pgk}-\mu_{pk}}{\sqrt{\operatorname{mean}_g(R_{pgk}-\mu_{pk})^2}+\varepsilon},\quad L=-\operatorname{mean}_{p,g,t}\min(e^{l_{pgt}}A_{pg},\operatorname{clip}(e^{l_{pgt}},1-c,1+c)A_{pg}).
$$

## 3. From equations to code

```python
shared_noise(base: Tensor, group_size: int) -> Tensor
```
```python
group_advantages(rewards: Tensor, eps: float=1e-8) -> Tensor
```
```python
dance_loss(log_ratios: Tensor, advantages: Tensor, clip: float) -> Tensor
```

Replicate one initial noise vector across each prompt group. Center and scale each reward channel within its own group, then sum channel advantages. Broadcast terminal advantages to transitions and apply the clipped objective.

## 4. Contract

shared_noise accepts base (P,D) and group_size>=2, returns (P,G,D). group_advantages accepts rewards (P,G,K), G>=2; population variance plus eps outside sqrt, sum K. dance_loss log_ratios (P,G,T), advantages (P,G), clip in (0,1), abs(log_ratio)<=40.

Noise and reward-derived advantages are frozen. Only current log ratios differentiate in the policy loss; each prompt is normalized independently.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

Two prompt groups share initial states but draw independent Gaussian actions. Two differently scaled quadratic reward channels train a scalar conditional policy for 70 updates.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/dancegrpo --impl problems/diffusion_rl/dancegrpo/solution.py
python -m paperleet.cli mutate diffusion_rl/dancegrpo
```

## 7. Extensions

Compare within-group reward variance under shared and independent initial noise.

[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) · Section 2.2 Eqs. (9)-(10), Initialization Noise; Section 2.3 combined rewards
