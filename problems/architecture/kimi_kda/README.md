# Kimi K3：有界衰减与 Delta 记忆

本题来自 [Kimi K3: Open Frontier Intelligence，v2，§2.1.1，公式 1/5](https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1)，固定采用 2026 年 8 月 7 日版本。只提取 Kimi Delta Attention 的记忆更新核心；[官方 K3 仓库](https://github.com/MoonshotAI/Kimi-K3)和 [FlashKDA](https://github.com/MoonshotAI/FlashKDA)可供进一步阅读。

## 你要实现的核心

### 返回值要求

#### `bounded_decay` — 需要实现

返回一个与 logits 同形状的浮点保留因子 Tensor `[B,T,H,K]`；B 为批大小，T 为序列长度，H 为头数，K 为键通道数。取值在 exp(lower_bound) 与 1 之间，返回因子本身而非其对数。保持 logits 的 dtype/device 及 logits、log_scale 的梯度，不修改输入。

#### `recurrent_kda` — 需要实现

返回浮点 Tensor 元组 `(output, final_state)`，形状为 `[B,T,H,V]` 和 `[B,H,K,V]`。B 为批大小，T 为 token 数，H 为头数，K/V 分别为键/值通道数，可不相等。每个输出读取该 token 更新后的状态；final_state 可用于继续扫描。T=0 时输出为 `[B,0,H,V]`，最终状态等于传入初始状态，初始状态为 None 则返回零状态。保持共同 dtype/device；非空扫描保留 q/k/v/alpha/beta/initial_state 的梯度，不 detach 返回状态或修改输入。

### 公式与约定

- `bounded_decay`：把每个 key 通道的 logit 转成保留系数。
- `recurrent_kda`：读取已经投影好的 q/k/v 与门值，返回各步输出和最终记忆。

令 $A$ 为每个头的 log-scale，$g_{\min}<0$：

$$g=g_{\min}\sigma(e^A z),\qquad \alpha=e^g.$$

每步依次执行遗忘、修正旧预测、读取新状态：

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1},\qquad
S_t=\bar S_t+\beta_t k_t(v_t-k_t^\top\bar S_t)^\top,\qquad o_t=S_t^\top q_t.
$$

直接累加 `beta * outer(k, v)` 会把同一个 key 的多次写入叠加；delta-rule 则先减去记忆对当前 key 的预测，再写入误差。

| 参数 / 输出 | 约定 |
|---|---|
| `logits`, `log_scale` | `[B,T,H,K]`, `[H]`；`lower_bound` 默认 −5 |
| `q`, `k`, `v` | `[B,T,H,K]`, `[B,T,H,K]`, `[B,T,H,V]`；允许 `K != V` |
| `alpha`, `beta` | `[B,T,H,K]`, `[B,T,H]`；已经是保留系数和写入强度，不是 logits |
| `initial_state` | 可选 `[B,H,K,V]`，未提供时从零开始 |
| 输出 / 最终状态 | `[B,T,H,V]` / `[B,H,K,V]`；当前 token 读取更新后的状态 |
| 空序列 | 输出时间维为 0，状态保持输入状态或默认零状态 |
| 数值范围 | 有限浮点输入，保持 dtype/device、梯度和原输入，支持非连续视图 |

本题不在 scan 中再次缩放 q 或归一化 k；它们已经由外部模型准备好。投影、短卷积、输出 RMSNorm、输出门和完整 KDA–MLA 网络不在范围内。这里状态按论文采用 key×value 排列，有些官方 kernel 会存转置。允许 alpha/beta 为 0 或 1，作为便于验证的解析极限；有限 sigmoid 参数通常只产生开区间内数值。

## Toy 实验

第一张图重绘论文 Figure 3(a)：固定 $A=0$、$g_{\min}=-5$，扫描 −10 到 10 的 logit，对比 K3 有界 log-decay 与 Kimi Linear 的负 softplus。标记为 **paper-derived**，不表示复现 GPU 加速比。

第二张图是单独标注的 **PaperLeet 教学改编**：key 0 写入 2，key 1 写入 7，再把 key 0 改写成 5，最后只保留 key 0 的四分之一且停止写入。解析输出为 `[2, 7, 5, 1.25]`。无需训练或下载；T3 实际调用你的函数，参考预览与个人运行结果分别显示。

## 验证内容

T0 检查接口、空序列和输入不变；T1 用直接矩阵公式独立计算；T2 检查门值、六类输入的梯度、分段续接、因果性、零写入和非连续输入；T3 对照解析目标验证两张图。本题不会用 CPU toy 声称 GPU kernel 性能。
