# 多头注意力

一个注意力头使用一个投影后的特征空间决定从哪些 token 读取信息。多个头可以从同一组 token 中读取不同的信息。本题实现 [Attention Is All You Need §3.2.2、Figure 2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2) 中的投影与组合流程。作者的原始实现位于 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)。

## 你要实现的核心模块

### 返回值要求

#### `multi_head_attention` — 需要实现

依次返回浮点 Tensor 元组 `(output, weights)`，形状为 `[B,Q,D]`、`[B,H,Q,K]`；B 为批大小，Q/K 为查询/键长度，D 为模型宽度，H=num_heads。output 已拼接各头并经过 w_o 投影；weights 保留每个头，不进行头平均。保持输入 dtype/device 及 query/key/value、四个投影矩阵的梯度。全遮挡查询行返回零，不修改输入。

#### `attention_kernel` — 已提供

返回浮点 Tensor 元组 `(output, weights)`，形状为 `[B,H,Q,Dv]`、`[B,H,Q,K]`；B 为批大小，H 为头数，Q/K 为查询/键长度，Dv 为值向量宽度。权重在可见键上求和为 1，全遮挡行的输出和权重为零。保持输入 dtype/device 和 q/k/v 的 autograd，全遮挡行梯度有限。此 helper 返回拼接/输出投影之前的逐头结果，不修改输入。

### 公式与约定

只实现 `multi_head_attention`。编辑器已提供稳定的缩放点积注意力函数 `attention_kernel`，因此本题可以独立于缩放点积注意力题完成。核心是输入投影、拆分多个头、传递掩码、拼接各头输出，以及最终的输出投影。

$$
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),\qquad
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O.
$$

| 输入 / 输出 | 约定 |
|---|---|
| `query`、`key`、`value` | `[B,Q,D]`、`[B,K,D]`、`[B,K,D]`；query 与 key 长度可以不同 |
| `w_q`、`w_k`、`w_v`、`w_o` | `[D,D]`；按论文约定使用 `x @ w`，与 `nn.Linear.weight` 的存储方向不同 |
| `num_heads` | 能整除 `D` 的正整数；每个头宽度为 `D/H` |
| 头的分配 | 投影结果中连续的特征切片依次对应各个头 |
| `mask` | 可广播到 `[B,H,Q,K]` 的可选布尔张量；`True` 表示允许关注 |
| 返回输出、权重 | `[B,Q,D]`、`[B,H,Q,K]`；保留每个头独立的权重 |
| 全部被屏蔽的行 | 权重和输出为零；该行的反向传播梯度也必须为有限的零 |
| 范围 | 输入有限且浮点 dtype、device 一致；不含 bias、dropout；保留梯度、不修改输入 |

明确维度顺序：先投影得到 `[B,Q,D]`，reshape 成 `[B,Q,H,D/H]`，再 transpose 成 `[B,H,Q,D/H]`。注意力结束后，必须先将 head 维换回去，再合并最后两个维度。直接 reshape 会把不同 query 的特征混在一起。Cross-attention 中，query 来自一个序列，key/value 来自另一个序列。

全屏蔽行的零输出约定是 PaperLeet 为 padding 场景补充的行为，提供的内核已经处理。残差连接和归一化属于 Transformer 的其他模块，不放在本函数内。

## Toy 实验

两个 query 通过两个头读取三个 token。第一个 query 的第一个头偏向 key 0，第二个头偏向 key 2。实验根据已知 logits 独立算出精确概率，再验证拼接结果和一个非恒等输出投影。图中会展示两个头各自关注的 key，以及输出特征与解析目标的对比。

**来源：** 这是根据论文多头算子设计的 PaperLeet 教学改编，不是论文原始实验。只用小型合成数据，无需训练或下载数据。T3 调用提交的函数，参考实现的预览会单独标注。

## 验证

T0 检查 cross-attention 接口及输入不被修改。T1 通过逐头独立投影与 PyTorch 注意力算子验证。T2 检查广播及逐头掩码、全屏蔽行、三个输入和四个投影矩阵的梯度、非连续张量、单头退化情况、因果隔离。T3 运行双头路由实验。常见错误包括 key 误用 query 投影、遗漏 mask、合并维度顺序错误，以及漏掉输出投影。
