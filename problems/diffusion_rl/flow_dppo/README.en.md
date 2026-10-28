# Flow-DPPO Exact Divergence and Directional Masks

## 1. Problem

Use exact Gaussian KL to block updates already moving farther from the old policy while retaining corrective directions.

Implement a fixed-covariance divergence mask and policy objective with optional reference KL, excluding predictive-mask variants.

## 2. Core equations

$$
D=\|\mu_\theta-\mu_{old}\|^2/(2s^2),\quad M=1-\mathbf1[D>\delta\ \land\ ((A>0\land l>0)\lor(A<0\land l<0))],\quad L=\operatorname{mean}[-M e^l A+\beta D_{ref}].
$$

## 3. From equations to code

```python
policy_kl(current: Tensor, old: Tensor, std: Tensor) -> Tensor
```
```python
flow_dppo_loss(log_ratios: Tensor, advantages: Tensor, old_kl: Tensor, reference_kl: Tensor, threshold: float, beta: float) -> Tensor
```

Measure policy movement with exact equal-covariance Gaussian KL. Block only when advantage and log-ratio signs agree and divergence strictly exceeds the threshold. Replace blocked logits before exponentiation to avoid zero-times-infinity; retain reference KL on every step.

## 4. Contract

policy_kl current/old (B,T,D), positive std (T,) returns (B,T). flow_dppo_loss log_ratios/old_kl/reference_kl equal (B,T), advantages (B,), threshold>=0,beta>=0. Old KL must be nonnegative; strict > threshold and >/< ratio=1; unmasked log ratios bounded to +/-40.

Only log_ratios and reference_kl differentiate in loss. Advantages and divergence decisions are frozen. The mask is zero only for favorable-direction updates beyond the divergence threshold; blocked ratios are replaced before exponentiation.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A scalar Gaussian policy draws 128 fresh actions per update and optimizes a quadratic reward with beta=0.1 reference KL for 65 iterations. Independent tests exercise nonzero old-policy divergence masks.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/flow_dppo --impl problems/diffusion_rl/flow_dppo/solution.py
python -m paperleet.cli mutate diffusion_rl/flow_dppo
```

## 7. Extensions

Compare PPO clipping and divergence thresholds under changes in action dimension.

[Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models](https://arxiv.org/abs/2606.11025) · Section 3.3 Eqs. (14), (17)-(18): Gaussian divergence and directional mask
