# Differential Transformer 差分注意力

实现 [Differential Transformer §2.1](https://arxiv.org/html/2410.05258v1#S2.SS1)（ICLR 2025 Oral）的核心。分别归一化两张注意力图，再相减：

$$A_1=\operatorname{softmax}(Q_1K_1^T/\sqrt d+M),\quad A_2=\operatorname{softmax}(Q_2K_2^T/\sqrt d+M),\quad O=(A_1-\lambda A_2)V.$$

因果模式中，允许位置的 `M=0`，未来位置为 `−∞`。沿 **key** 维度做 softmax，**两条分支**都需要掩码。

同时实现标量参数化：

$$\lambda=\exp(\lambda_{q1}^T\lambda_{k1})-\exp(\lambda_{q2}^T\lambda_{k2})+\lambda_{init}.$$

## 接口

### 返回值要求

#### `differential_lambda` — 需要实现

返回一个零维浮点 Tensor（`shape == ()`），表示标量系数 lambda；不是 Python float 或长度为 1 的向量。数值不限制在 [0,1]。保持四个参数向量的 dtype/device 及梯度，不修改输入。

#### `differential_attention` — 需要实现

返回一个浮点 Tensor `[B,H,T,2D]`：B 为批大小，H 为头数，T 为序列长度，D 为 Q/K 宽度；形状与 value 相同。它是每头归一化和输出投影之前的差分值聚合，不返回注意力权重元组。保持输入 dtype/device，以及两组 Q/K、value 和 Tensor 系数的梯度，不修改输入。

### 公式与约定

- `differential_lambda(q1, k1, q2, k2, lambda_init=0.2)` 输入相同且非空的参数向量 `[Dλ]`，返回标量张量。这些是 lambda 参数，与注意力 Q/K 张量不同。调用方提供层相关初值；原文从第 1 层起使用 `0.8 - 0.6 exp(-0.3(l-1))`。
- `differential_attention(q1, k1, q2, k2, value, coefficient, causal=True)` 的 Q/K 为 `[B,H,T,D]`，V 为 `[B,H,T,2D]`，返回 `[B,H,T,2D]`。系数可为有限 Python 标量或标量张量，后者必须保留梯度。

输入为有限、非空、可以非连续的 CPU float32/float64 张量。保留 dtype、device、梯度与输入值；测试限制 lambda 指数在有限范围内。脚手架已提供形状和标量校验。不使用 dropout。本题不实现逐头归一化及其固定缩放、投影层或完整模型。

差分图的每行和为 `1−λ`，元素可以为负，不是概率分布。`λ=0` 时退化为普通注意力；两图相同且 `λ=1` 时输出为零。

## Toy 实验

**教学改编。** 四个 token 具有可区分的信号，每个 value 叠加相同偏移。用你的注意力将 lambda 从 0 扫到 1，测量共同偏移的响应：恰为该偏移的 `(1−λ)` 倍。另一条曲线显示不同注意力图保留的信号。这个受控性质不意味着任意噪声都会下降。

预览使用参考实现，T3 使用你的函数。独立标量 softmax 和解析向量-Jacobian 积检验输出与全部 Q/K/V/lambda 梯度。[作者实现](https://aka.ms/Diff-Transformer)。
