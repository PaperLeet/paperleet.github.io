# Kimi K3：Block Attention Residuals

把“网络深度”也作为 attention 的选择轴。本题提取 [Kimi K3 v2，§2.2，公式 8–10](https://arxiv.org/html/2607.24653v2#S2.SS2) 的核心聚合算子，并对照作者的 [Attention-Residuals 官方伪代码](https://github.com/MoonshotAI/Attention-Residuals/blob/master/README.md)。K3 报告固定为 2026 年 8 月 7 日版本。

## 你要实现的核心

### 返回值要求

#### `assemble_sources` — 需要实现

返回一个浮点 Tensor `[N,B,T,D]`：N 为深度来源数，B 为批大小，T 为 token 数，D 为特征宽度。来源顺序为 embedding、原顺序的 completed_blocks、存在时的 partial_block；`N=1+len(completed_blocks)+(partial_block is not None)`。显式传入的零块仍占一个来源。保持 dtype/device 及各来源的梯度，不修改输入 Tensor 或调用方列表。

#### `depth_attention` — 需要实现

返回由两个浮点 Tensor 组成的 Python 元组 `(mixed, weights)`。`mixed[B,T,D]` 是原始来源值的聚合，`weights[N,B,T]` 是深度来源概率，对每个批样本/token 沿 N 求和为 1。N 为来源数，B 为批大小，T 为 token 数，D 为特征宽度。保持 sources 的 dtype/device，以及 sources、query、norm_weight 的梯度，不修改输入。

### 公式与约定

`assemble_sources` 组装 token embedding、已完成块的输出和、当前块的部分输出和。`depth_attention` 用可学习 pseudo-query 从这些深度来源中选择信息。完整网络和各块内部的输出累加由外部提供，不需要复现整个 Transformer。

对来源 $v_i$、查询 $w$ 和 RMSNorm 权重 $\gamma$：

$$k_i=\frac{v_i}{\sqrt{\operatorname{mean}(v_i^2)+\epsilon}}\odot\gamma,
\qquad a_i=\operatorname{softmax}_{i}(w^\top k_i),
\qquad h=\sum_i a_i v_i.$$

只有 key 经过归一化，value 保持原始幅度。分数不除以 Transformer 常见的 $\sqrt D$。

| 参数 / 输出 | 约定 |
|---|---|
| `embedding` | `[B,T,D]`，永远是第一个来源 |
| `completed_blocks` | `[B,T,D]` 的列表，不包含 embedding；每个元素已经是一个块的输出和 |
| `partial_block` | 可选 `[B,T,D]`；`None` 表示不存在，显式零张量仍是一个真实来源 |
| `sources` | `[N,B,T,D]`，第一维为深度来源 |
| `query`, `norm_weight` | `[D]`，当前层共享于各 token 的可学习向量 |
| `eps` | 正数，放在 RMS 平方根内部，默认 `1e-6` |
| 输出 / 权重 | `[B,T,D]` / `[N,B,T]`；每个 token 独立沿深度归一化 |
| 数值范围 | 有限同 dtype/device 浮点输入；保留列表、张量、梯度，支持非连续输入 |

新块第一层还没有 partial sum，不能加一个零占位符，否则会改变 softmax 分母。有实际 partial sum 时，即使值刚好全零也必须保留。completed blocks 已经存好各层输出的和，不要再给每个块加一次 embedding。

## Toy 实验

提供三个合成来源：embedding `[1,0]`、已完成块 `[-2,0]`、当前块 `[0,3]`。把 pseudo-query 从 `[0,0]` 扫描到 `[6,0]`，展示各来源的权重与输出到有效 embedding 的均方误差。独立标量计算给出解析 softmax 目标。

**来源标注：PaperLeet 教学改编。** 实验根据论文算子和官方伪代码构造，只扫描 query，不训练模型，也不冒称作者原实验。T3 会实际调用你实现的两个函数，个人运行曲线与参考预览分别显示。

## 验证内容

T0 检查来源顺序、接口和输入不变；T1 用 PyTorch RMSNorm 和逐 token 计算建立独立 oracle；T2 检查缺省与零 partial block、零 query、所有参数梯度、深度重排、非连续输入和大 logits；T3 验证合成的深度选择行为。
