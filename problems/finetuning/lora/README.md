# LoRA 低秩更新与冻结基座前向

实现缩放低秩更新并冻结基座参数。

论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2)。定位：Section 4.1, Eq. (3): W0 + BA and alpha/r scaling；Section 4.1: random A, zero B initialization and merged inference。

## 精选核心

### 返回值要求

#### `lora_delta` — 需要实现

从 a `[r,in_features]`、b `[out_features,r]` 返回一个浮点 Tensor `[out_features,in_features]`，表示经缩放的低秩权重增量。保持共同 dtype/device 及两个 adapter 的梯度。仅返回增量矩阵，不合并进或修改基座权重。

#### `lora_linear` — 需要实现

对 x `[...,in_features]` 返回一个浮点 Tensor `[...,out_features]`，保留所有前导批/token 维度。其值包含基座线性输出和 adapter 贡献，可选 bias 仅加入一次。保持 x 的 dtype/device 及 x、a、b 的梯度；weight 和可选 bias 即使需要梯度也应冻结。不修改输入，也不返回更新后的参数。

### 公式与约定

实现线性层的两个函数；初始化与 rank-2 回归训练循环已提供。

$$
y=xW_0^\top+\frac\alpha r xA^\top B^\top
$$

- `lora_delta(a: Tensor, b: Tensor, alpha: float=1.0) -> Tensor`
- `lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, bias: Tensor | None=None) -> Tensor`

## 接口与实验

实现 `lora_delta(A,B,alpha)` 与 `lora_linear(x,W,A,B,alpha,bias)`。形状为 `A[r,in]`、`B[out,r]`、`W[out,in]`、`x[...,in]`。增量为 `(alpha/r) * B @ A`。前向等于冻结基座线性映射加低秩更新。即使输入基座权重与 bias 的 `requires_grad=True`，也不能向它们传播梯度；`x`、`A`、`B` 的梯度必须保留。

训练脚手架随机初始化 A、零初始化 B。因此初始输出不变，B 有学习信号而 A 的首次梯度为零。对整个基座计算使用 `no_grad()` 会错误删除部分 x 梯度。将 delta 加到 W 即可合并，测试比较合并前后结果。本模块不包含 dropout。

Toy 冻结一个 6×8 映射，用 28 个 adapter 参数拟合已知 rank-2 变化。平台提供输入、优化器和 300 步 CPU 训练。通过表示恢复这个可控目标，不是复现完整语言模型指标。

## Toy 来源

**PaperLeet 教学改编。** PaperLeet rank-2 合成回归，不是语言模型基准。 判题 harness 调用本次提交的函数，保存由其产生的指标与曲线。Lab 静态预览来自参考实现，Run/Submit 的产物来自你的代码。`experiment.py` 与 `tests/` 已提供实验和判题脚手架。

## 来源与验证

[微软 LoRA 官方实现](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)。题目实现与合成数据由 PaperLeet 编写；论文链接固定公式依据的版本。

T0 验证接口，T1 对照参考值，T2 检查数学性质与梯度，T3 用你的实现运行 toy。在在线编辑器补全 starter 后 Run 或 Submit，无需部署完整大模型训练系统。
