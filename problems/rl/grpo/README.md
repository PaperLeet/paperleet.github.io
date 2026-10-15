# GRPO 组相对策略目标

## 1. 这篇论文解决什么

GRPO 对同一 prompt 采样一组回答，用相对奖励代替 value 模型，减少语言模型强化学习中的 critic 成本。[DeepSeekMath 第 4.1 节](https://arxiv.org/html/2402.03300v3#S4.SS1) 给出目标与组内估计。

本模块目标：实现组内相对 advantage 与按回答聚合的截断策略目标。 只实现 advantage 和 loss 两个函数；mask 辅助、诊断量与采样实验已提供。 完成后记为这篇论文的一个核心模块。

## 2. 核心公式与题面

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `grpo_advantages` · 需要实现

返回一个浮点 Tensor `advantages`，形状 `[B * G]`，保持原本组相邻的顺序（`B` 个问题，每组 `G = group_size` 个回答），保留 `rewards` 的 dtype/device。每组减均值；仅在 `scale_std=True` 时除以样本标准差加 `eps`。返回展平的向量，不是 `[B, G]` 矩阵或标量。

#### `grpo_loss` · 需要实现

返回 `(loss, stats)`，即“标量 Tensor、字典”的二元组。`loss` 形状为 `()`：对于 `[N, L]` 输入（`N` 个回答、`L` 个 token 位置），先对每个回答的有效 token 平均，再对全部回答平均；空回答贡献零。保留 `logp` 的梯度，旧/参考分数及 advantage 固定。字典字段为 `kl_mean`、`clipfrac`、`ratio_mean`，均为已 detach、形状 `()` 的浮点标量 Tensor，按全部有效 token 平均。`clipfrac` 统计 `abs(ratio - 1) > clip_eps` 的比例；没有有效 token 时三者均为零。

#### `masked_logprobs` · 已提供

返回 `(masked_logp, masked_old_logp, masked_ref_logp)`，与三个分数输入顺序一致。每个 Tensor 都保留对应输入的 `[N, L]` 形状、dtype 和 device，将 padding 位置置零，其余位置不变。这个辅助函数保留有效位置的梯度路径，旧/参考分数由调用它的 loss 函数另行冻结。

#### `token_statistics` · 已提供

对 `[N, L]` token 数组返回 `dict[str, Tensor]`。字典字段为 `kl_mean`、`clipfrac`、`ratio_mean`，均为已 detach、形状 `()` 的浮点标量 Tensor，按全部有效 token 平均。`clipfrac` 统计 `abs(ratio - 1) > clip_eps` 的比例；没有有效 token 时三者均为零。

### 公式与约定

实现 `grpo_advantages` 和 `grpo_loss`。奖励 `[B*G]` 中，同一 prompt 的 $G$ 个回答必须相邻，$G\ge2$ 且整除奖励数。组内均值 $\bar r_g$ 和样本标准差 $s_g$ 用来计算：

$$
\hat A_i=\frac{r_i-\bar r_g}{s_g+\varepsilon},\qquad
\rho_{i,t}=e^{\log\pi_{i,t}-\log\pi^{old}_{i,t}},\quad
k3_{i,t}=e^{d_{i,t}}-d_{i,t}-1,
\quad d=\log\pi^{ref}-\log\pi.
$$

`scale_std=False` 时只减均值；默认 $\varepsilon=10^{-4}$，标准差采用 `correction=1`。三个 log-prob 与 mask 的 shape 是 `[N,L]`，$N=B G$；advantage 是 `[N]`，先扩展末维以广播。

$$
L=-\frac1N\sum_i\frac1{\sum_t m_{i,t}}\sum_t m_{i,t}
\left[\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-c,1+c)\hat A_i)-\beta k3_{i,t}\right].
$$

默认 $c=0.2,\beta=0.04$。全 padding 回答贡献零；所有行仍计入回答平均。返回标量 loss 和 `kl_mean, clipfrac, ratio_mean`。三个诊断量对有效 token 平均并 detach，其中 clipfrac 统计 $|\rho-1|>c$。

## 3. 从公式到代码

```text
rewards [B*G] → reshape [B,G] → 减组均值，可选除组 std → 展平
logp/old/ref [N,L] → 冻结 old/ref → 构造 ratio 与 k3
advantages [N] → [N,1] → 与每个 token 的 ratio 相乘
逐 token 悲观目标乘 mask → 每行求和/有效长度 → 回答平均
```

starter 已提供 `masked_logprobs(logp, old_logp, ref_logp, mask)` 与 `token_statistics(ratio, kl, mask, clip_eps)`。前者将 padding 位置置零，后者返回所需的三个 detach 诊断量。可直接调用，集中实现组相对 advantage、clipped surrogate 和按回答聚合。

先隔离 padding，再计算指数，可避免无效位置的极大 log-prob 差造成 `inf*0`。旧策略与参考策略是训练时的固定快照，即使调用者误传了可求导张量也必须切断它们的梯度。

## 4. 本题约定

| 项目 | 本题 | 常见实现 |
|---|---|---|
| std | 组内样本标准差，加 $10^{-4}$ | [TRL v0.23.1](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py) 的 group 模式相同 |
| 聚合 | 先每回答 token 平均，再回答平均 | DAPO 使用有效 token 聚合 |
| KL | k3 直接加入 loss | 不先从奖励中扣 KL |
| 诊断 | 有效 token 比率的双尾越界比例 | 部分库统计实际被截断的有利方向 |

$e^d-d-1\ge0$ 保证单个 token 的 KL 项非负；与 KL 的期望对应还需要合适的采样分布。T3 的 toy 策略包含 EOS，用八组、每组八条回答，在 150 步内学习目标串 `abc` 的前三位匹配奖励。

同一问题的所有奖励都相等时，中心化后全部为零，不应产生奖励驱动的策略更新。若开启 KL 项，策略仍可能被拉向参考模型；因此调试零 advantage 时，要区分奖励梯度与正则项梯度。

## 5. 交互可视化

[打开组内 advantage](#viz-group_advantage)：修改单个奖励，观察同组回答如何共同变化；切换 std 归一化查看量纲的影响。

[打开 token 梯度热图](#viz-token_weight_heatmap)：调节回答长度，对照 GRPO、DAPO 和 Dr. GRPO 的逐 token 权重。训练曲线同时记录奖励与回答长度。

## 6. 怎么运行

```bash
paperleet init rl/grpo
paperleet test rl/grpo
```

编辑 `workspace/rl/grpo.py`。先验证归一化与 padding，再检查旧量无梯度，最后运行带 EOS 的策略训练。

## 7. 延伸

下一题 [Dr. GRPO](../dr_grpo/README.md) 只更改 advantage 缩放和聚合分母，直接检验长度偏置与难度偏置。KL 估计器的推导可参考 [Schulman 2020](http://joschu.net/blog/kl-approx.html)。
