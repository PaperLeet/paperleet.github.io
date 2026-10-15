# 在线 Softmax 与分块注意力

[FlashAttention](https://arxiv.org/abs/2205.14135) 通过减少 GPU 内存层级之间的数据传输来计算精确注意力。其数值核心是在线 softmax：逐块合并结果，无需一次生成完整注意力矩阵。阅读 §3.1 与算法 1；作者生产级 kernel 见 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。

## 核心任务

### 返回值要求

#### `online_softmax_update` — 需要实现

按固定顺序返回浮点 Tensor 的 Python 元组 `(new_max, new_normalizer, new_numerator)`，形状分别为 `[Q]`、`[Q]`、`[Q,Dv]`；Q 为查询行数，Dv 为值向量宽度。三项表示已处理所有键分块的行最大值、未归一化指数总量、未归一化加权值之和；第三项尚不是归一化注意力输出。保持 dtype/device 和 autograd，不原地修改输入状态。

#### `block_attention` — 已提供

返回浮点 Tensor 的 Python 元组 `(output, logsumexp)`，形状为 `[Q,Dv]` 和 `[Q]`；Q 为总查询长度，Dv 为值向量宽度。output 是归一化注意力结果；logsumexp 是每个查询对全部键的缩放分数的自然对数 sum-exp，不是注意力权重矩阵。保持 q/k/v 的 dtype/device 和梯度路径，包括 logsumexp 的梯度，不修改输入。

### 公式与约定

只需实现 `online_softmax_update`。提供的 `block_attention` 负责 query/key 双向分块、除以 `sqrt(D)` 和最终归一化。返回新的行最大值、归一化常数和**尚未归一化**的 value 加权和。

给定旧状态 `(m,l,u)`、新 score 分块 `S` 及对应 value `V`：

$$
m'=\max(m,\max_j S_j),\quad \alpha=e^{m-m'},\quad P_j=e^{S_j-m'},
$$
$$
l'=\alpha l+\sum_jP_j,\qquad u'=\alpha u+PV.
$$

处理完所有分块后，输出 `u/l`，log-sum-exp 为 `m + log(l)`。累计最大值变化时，旧概率质量与旧 value 加权和**都要缩放**。

| 项目 | 约定 |
|---|---|
| `scores`、`values` | `[Q,K]`、`[K,Dv]`，score 有限，每块至少一个 key |
| 状态 | `row_max [Q]`、`normalizer [Q]`、`numerator [Q,Dv]` |
| 初始状态 | 负无穷、零、零 |
| 外层输入 | `q [Nq,D]`、`k [Nk,D]`、`v [Nk,Dv]`，块大小为正整数 |
| 外层输出 | attention `[Nq,Dv]` 与 log-sum-exp `[Nq]` |
| 梯度 | 保留所有分块的计算图，不修改传入状态 |

本题使用有限 score，不含 mask 或 dropout。因果与全屏蔽语义由 Transformer 模块讲解。

## 可手算例子

两个单 key 分块的 score 为 `0`、`log(3)`，对应 value 为 `2`、`6`。未归一化质量为 1、3，最终输出是 `(1*2 + 3*6)/4 = 5`，log-sum-exp 为 `log(4)`。

## Toy 实验与范围

提供的 37 token 实验使用块大小 1、3、8、16、37、64，与独立计算的稠密 attention 对比。报告包括输出和 LSE 最大误差，以及最大 score 分块的理论元素数。末尾不足一块的情况也必须正确。

**来源：** 根据算法 1 编写的 PaperLeet CPU 教学改编，使用合成张量，不是原论文性能实验，也不复现 GPU 加速。自动求导可能保留中间分块，理论分块元素数不代表实测训练内存节省。融合 kernel、GPU IO 调度和重新计算的反向传播不在本题范围内。

## 验证

T0 覆盖矩形输入与边界分块；T1 检查可手算例子；T2 验证分块不变性、极端分数稳定性、状态可复用性、与稠密实现的梯度一致性；T3 使用提交的递推函数完成六种块大小的实验并生成结果表。常见错误是漏缩放某个状态、用当前块最大值覆盖累计最大值或切断旧分块梯度。
