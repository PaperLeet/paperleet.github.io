# DDPO 高斯转移密度与逐步 PPO

## 1. 问题

计算高斯去噪转移的联合 log 密度，并实现终点优势广播的 clipped PPO 损失。

使用固定方差高斯转移实现 DDPO 的策略目标，包含在线小型 bandit；不加载扩散骨干。

## 2. 核心公式

$$
\ell=-\frac12\sum_d[(a_d-\mu_d)^2/s^2+2\log s+\log(2\pi)],\quad L=-\operatorname{mean}_{b,t}\min(e^{\ell-\ell_{old}}A_b,\operatorname{clip}(e^{\ell-\ell_{old}},1-e,1+e)A_b).
$$

## 3. 公式到代码

```python
gaussian_log_prob(mean: Tensor, actions: Tensor, std: Tensor) -> Tensor
```
```python
ddpo_loss(current: Tensor, old: Tensor, advantages: Tensor, clip: float) -> Tensor
```

每个转移的坐标共同构成一个动作，因此先对特征维求 log 密度之和。旧策略和采样动作保持固定。逐步概率比乘终点优势，对正负优势分别由 min 选择保守分支。

## 4. 本题约定

gaussian_log_prob: means/actions (B,T,D), std (T,) strictly positive; return (B,T) summing D. ddpo_loss: current/old (B,T), advantages (B,), 0<clip<1. Average B,T.

Gaussian actions and std are detached; current means differentiate. PPO old log probabilities and advantages are detached. Very large unfavorable log ratios may overflow; finite tested ratios stay within the declared numerical domain |current-old|<=40.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A scalar Gaussian policy with standard deviation 0.6 draws 128 fresh actions per update; 60 PPO gradient updates optimize a quadratic terminal reward toward target 1.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/ddpo --impl problems/diffusion_rl/ddpo/solution.py
python -m paperleet.cli mutate diffusion_rl/ddpo
```

## 7. 延伸

加入按 prompt 维护的奖励统计量，而不是跨 prompt 共享一个基线。

[Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) · Section 4.3 Eq. (6), importance-sampled DDPO; Appendix D.1
