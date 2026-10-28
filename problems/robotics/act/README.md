# ACT 动作块时间集成与风格正则

## 1. 问题

实现对当前动作的重叠预测集成，以及条件VAE的标准正态KL。

实现ACT的两个张量核心；输入已经对齐到同一执行时刻，省略视觉Transformer、动作块网络和遥操作系统。

## 2. 核心公式

$$
\bar a=\frac{\sum_{i=0}^{K-1}e^{-ki}a_i}{\sum_i e^{-ki}},\quad L_{KL}=\frac1{2B}\sum_{b,d}(\mu_{bd}^2+e^{\ell_{bd}}-1-\ell_{bd}).
$$

## 3. 公式到代码

- `temporal_ensemble(actions: Tensor, valid: Tensor, decay: float) -> Tensor`
- `style_kl(mu: Tensor, logvar: Tensor) -> Tensor`

官方实现按旧到新索引使用负指数；因此正decay给较早预测更大权重。显式valid替代源码的全零哨兵，避免把合法零动作丢掉。KL中的logvar是对数方差。

## 4. 本题约定

actions为(K,D)，valid为(K,)布尔，至少一条有效；有效预测按生成时间从旧到新排列，先去padding再按0..Kvalid-1赋权，k>=0。mu/logvar为(B,Z)，KL按潜变量求和、按batch平均。保留浮点输入梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

对64个固定正弦目标构造五份交替误差预测，比较时间集成与最新单次预测的MSE；同时用KL梯度将风格均值移向零。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/act --impl problems/robotics/act/starter.py
paperleet mutate robotics/act
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) — Temporal ensembling at inference; official imitate_episodes.py exp_weights and CVAE policy KL
- [Verified source](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)
- [Verified source](https://tonyzhaozh.github.io/aloha/)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
