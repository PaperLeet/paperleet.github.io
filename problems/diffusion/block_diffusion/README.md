# Block Diffusion 分块注意力

## 1. 题目目标

[Block Diffusion](https://arxiv.org/html/2503.09573v2)（ICLR 2025 Oral）在先前干净块的条件下，对当前块去噪。向量化训练把输入拼成 **`[噪声 token，干净 token]`**。实现专用掩码，并把它应用到已投影的 Q/K/V。

## 2. 接口与返回值

### 返回值要求

#### `block_diffusion_mask`（需实现）

返回一个 `torch.bool` Tensor，形状 `[2L, 2L]`，`L=length`。行对应 query，列对应 key，两轴均按「噪声 L 个、干净 L 个」排列；`True` 表示允许关注。位于指定 `device`（本题默认 CPU），不带梯度；不是浮点加性掩码。

#### `block_diffusion_attention`（需实现）

返回注意力聚合后的 Tensor，形状 `[B, H, 2L, Dv]`，与 `value` 相同：B 为批大小，H 为头数，L 为每个流的 token 数，Dv 为 value 宽度。返回的是输出特征，不是 `[B,H,2L,2L]` 的权重矩阵。保留输入 float32/float64 dtype、device，以及对 Q/K/V 的梯度；不修改输入。

### 公式与接口说明

序列长度为 `L`，局部块编号为 `b(i)=floor(i/block_size)`。附录 B.6 给出的 `2L × 2L` 允许位置矩阵为：

$$M=\begin{bmatrix}M_{BD}&M_{OBC}\\0&M_{BC}\end{bmatrix},\quad
M_{BD}[i,j]=[b(i)=b(j)],\quad M_{OBC}[i,j]=[b(j)<b(i)],\quad M_{BC}[i,j]=[b(j)\le b(i)].$$

行是 query，列是 key。噪声 query 可见**自己整个噪声块**和**严格更早的干净块**。干净 query 可见自己与之前的干净块，始终不能看到噪声 token。如果噪声 query 看到自己的干净块，就泄漏了训练目标。

### 接口

- `block_diffusion_mask(length, block_size, device=None)` 返回布尔 `[2L,2L]` 张量；`True` 代表允许。两个尺寸均为正 Python 整数。
- `block_diffusion_attention(query, key, value, block_size)` 输入 Q/K `[B,H,2L,D]`、V `[B,H,2L,Dv]`，返回 V 的形状。点积分数除以 `sqrt(D)`，禁止位置设为 `−∞`，沿 key 维度归一化。

输入为有限、非空、可以非连续的 CPU float32/float64 张量。保留 dtype、device、梯度和输入值。脚手架提供校验；每个 query 至少有一个合法 key。支持最后不足整块的输入是本题的**教学扩展**，原文主要记号假定等长完整块。`block_size≥L` 时只有一个块。

本题只实现训练期注意力依赖，不实现离散加噪、NELBO、采样器、KV cache 或 GPU kernel。

## Toy：原文六 token 示例

**基于原文的改编。** 附录 B.6 Figure 3 使用 `L=6`、块大小 2。我们独立重算该掩码，再加入微型确定性注意力，逐一扰动每个 value token。曲线展示合法 key 数量及两个干净 token 的实际影响。禁止的影响必须为零，合法影响必须可观察。

例如噪声 query 2 只能读取拼接位置 `[2,3,6,7]`，干净 query 6 读取 `[6,7]`。预览使用参考实现，T3 使用你提交的掩码和注意力。数值 oracle 按每个块的条件上下文分别计算。[作者代码](https://github.com/kuleshov-group/bd3lms)。
