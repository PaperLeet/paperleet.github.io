# Flow-DPPO 精确散度与方向性掩码

## 1. 问题

用精确高斯 KL 限制已经远离旧策略的更新，同时保留朝旧策略恢复的方向。

实现固定协方差的散度掩码与策略目标，包含可选 reference KL；不实现附录中的预测式掩码。

## 2. 核心公式

$$
D=\|\mu_\theta-\mu_{old}\|^2/(2s^2),\quad M=1-\mathbf1[D>\delta\ \land\ ((A>0\land l>0)\lor(A<0\land l<0))],\quad L=\operatorname{mean}[-M e^l A+\beta D_{ref}].
$$

## 3. 公式到代码

```python
policy_kl(current: Tensor, old: Tensor, std: Tensor) -> Tensor
```
```python
flow_dppo_loss(log_ratios: Tensor, advantages: Tensor, old_kl: Tensor, reference_kl: Tensor, threshold: float, beta: float) -> Tensor
```

先用同方差高斯闭式 KL 测量策略分布变化。只在优势与 log 比率方向一致且散度严格超过阈值时屏蔽。屏蔽后的比率在 exponentiation 之前替换，防止零乘无穷大。reference KL 始终参与。

## 4. 本题约定

policy_kl current/old (B,T,D), positive std (T,) returns (B,T). flow_dppo_loss log_ratios/old_kl/reference_kl equal (B,T), advantages (B,), threshold>=0,beta>=0. Old KL must be nonnegative; strict > threshold and >/< ratio=1; unmasked log ratios bounded to +/-40.

Only log_ratios and reference_kl differentiate in loss. Advantages and divergence decisions are frozen. The mask is zero only for favorable-direction updates beyond the divergence threshold; blocked ratios are replaced before exponentiation.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A scalar Gaussian policy draws 128 fresh actions per update and optimizes a quadratic reward with beta=0.1 reference KL for 65 iterations. Independent tests exercise nonzero old-policy divergence masks.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/flow_dppo --impl problems/diffusion_rl/flow_dppo/solution.py
python -m paperleet.cli mutate diffusion_rl/flow_dppo
```

## 7. 延伸

对比 PPO ratio clipping 与散度阈值对动作维度变化的敏感度。

[Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models](https://arxiv.org/abs/2606.11025) · Section 3.3 Eqs. (14), (17)-(18): Gaussian divergence and directional mask
