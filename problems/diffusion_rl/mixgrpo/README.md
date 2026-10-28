# MixGRPO 滑动随机窗口与局部策略目标

## 1. 问题

实现随训练推进的 SDE 窗口、窗口外 ODE 转移，以及仅对窗口内步骤求均值的策略目标。

使用论文的基础混合 ODE-SDE 形式；不包含后续 CPS 扩展或高阶求解器。

## 2. 核心公式

$$
l=\min(\lfloor m/\tau\rfloor s,T-w),\quad W=\{l,\ldots,l+w-1\}.\quad L=-\operatorname{mean}_{b,t\in W}\min(e^{\ell_{bt}}A_b,\operatorname{clip}(e^{\ell_{bt}},1-c,1+c)A_b).
$$

## 3. 公式到代码

```python
window_start(update: int, total: int, width: int, interval: int, stride: int) -> int
```
```python
mixed_transition(state: Tensor, velocity: Tensor, time: float, next_time: float, noise_level: float, stochastic: bool) -> tuple[Tensor,Tensor]
```
```python
window_loss(log_ratios: Tensor, advantages: Tensor, start: int, width: int, clip: float) -> Tensor
```

按已完成更新次数推进窗口，边界到达末端后保持。ODE 步只用速度积分；SDE 步加上相应 score 修正与标准差。损失只切出窗口，按窗口长度而不是总轨迹长度平均。

## 4. 本题约定

window_start uses zero-based update index, total>=width>=1, stride>=1, interval>=1; saturates at total-width. mixed_transition: (B,D) state/velocity, 0<=next_time<time<1, noise_level>=0; returns mean and scalar std. window_loss: (B,T) log ratios, (B,) advantages, valid start/width, |selected ratios|<=40.

Rollout states are detached, velocity remains differentiable. Advantages are frozen; loss gradients are zero outside the selected window.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A four-transition scalar flow uses a moving two-step stochastic window. Fresh Gaussian paths and terminal quadratic rewards update one velocity parameter for 60 iterations.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/mixgrpo --impl problems/diffusion_rl/mixgrpo/solution.py
python -m paperleet.cli mutate diffusion_rl/mixgrpo
```

## 7. 延伸

在相同随机窗口上替换高阶 ODE 求解器并比较模型调用数。

[MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802) · Section 3.1 mixed ODE-SDE; Algorithm 1; Section 3.2 Eq. (11)
