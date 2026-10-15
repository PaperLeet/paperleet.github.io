# 缩放点积注意力

Transformer 根据 query 与 key 的匹配程度，从 value 中汇聚信息。本题实现 [Attention Is All You Need §3.2.1、公式 (1)](https://arxiv.org/abs/1706.03762) 的核心算子。作者原始实现见 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)。

## 核心任务

### 返回值要求

#### `scaled_dot_product_attention` — 需要实现

返回 Python 元组 `(output, weights)`，两项均为浮点 Tensor：`output[B,H,Q,Dv]` 是值向量的聚合结果，`weights[B,H,Q,K]` 是注意力概率。B 为批大小，H 为头数，Q/K 为查询/键长度，Dv 为值向量宽度。有可见键的每行概率和为 1；全遮挡行的权重和输出均为精确零。保持输入 dtype/device 及 q/k/v 的梯度；全遮挡行梯度为有限零，不修改输入。

#### `causal_mask` — 已提供

返回一个 `torch.bool` Tensor `[Q,K]`，Q=query_length、K=key_length；行为查询，列为键。键下标不大于查询下标时为 True，采用左上对齐。使用指定 device；可见性掩码不带梯度。

### 公式与约定

只需实现 `scaled_dot_product_attention`。`causal_mask` 与实验循环已经提供。输入已经完成投影并拆分 head；投影层、位置编码、前馈网络和完整语言模型训练不属于本题范围。

$$
S=QK^\top/\sqrt{d_k},\qquad A=\operatorname{softmax}_{\mathrm{keys}}(S+M),\qquad O=AV.
$$

可见位置的 mask 贡献为 0，不可见位置为负无穷。沿 **key 轴**分别对每个 query、每个 head 做稳定 softmax。返回输出 `O` 和权重 `A`，方便查看信息来自哪些位置。

| 输入或输出 | 约定 |
|---|---|
| `q`、`k`、`v` | `[B,H,Q,D]`、`[B,H,K,D]`、`[B,H,K,Dv]` |
| `mask` | 可广播至 `[B,H,Q,K]` 的布尔张量，`True` 表示可见 |
| 输出、权重 | `[B,H,Q,Dv]`、`[B,H,Q,K]`，保留精度与设备 |
| 全部屏蔽的 query | 输出和权重严格为 0，梯度有限且为 0 |
| 因果对齐 | 左上对齐：仅允许 `j <= i` 的 key |
| 范围 | 有限输入，无 dropout，允许 `D` 与 `Dv` 不同 |

全屏蔽行需要单独处理：直接对一整行负无穷做 softmax 会产生 NaN。先把这种行换成有限占位 logits，再将屏蔽位置的权重清零。这项 padding 约定是 PaperLeet 对论文公式的工程补充。

## 小例子

query 为 `[2,0,0,0]`，两个 key 为 `[1,0,0,0]`、`[0,0,0,0]` 时，缩放后的分数是 `[1,0]`，权重约为 `[0.7311,0.2689]`。若 mask 为 `[True,False]`，权重应为 `[1,0]`。

## Toy 实验

提供的**前缀平均**实验令全部 query/key 为零，再加入因果 mask。每个 query 的输出应恰好等于它目前见过的 value 的均值。把未来 value 增加 1000，前面的输出仍应不变。报告包含注意力矩阵、实际与期望前缀均值、未来信息泄漏量。

**来源：** PaperLeet 根据论文注意力与 mask 公式设计的教学改编，并非原论文实验。只用六个标量，不训练模型、不下载数据。T3 调用实际提交代码；实验室里的参考预览单独标记。

## 验证

T0 验证形状与输入未被修改。T1 使用可手算的两个 key。T2 验证广播 mask、空行、排列与平移不变性、独立梯度和因果隔离。T3 运行前缀平均实验。常见错误是遗漏平方根缩放、忽略 mask 或沿 query 轴归一化。

在编辑器中实现核心函数，再运行四层测试。完成本题表示掌握选定的 attention 核心，不等于复现完整 Transformer。
