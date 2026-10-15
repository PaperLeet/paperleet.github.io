# 18. Drifting：成对漂移场与冻结目标

## 1. 题目目标

实现距离核漂移场及 stop-gradient 目标，训练一次前向即可采样的生成器。

## 2. 接口与返回值

### 返回值要求

#### `drift_field`（需实现）

返回漂移向量 Tensor `[N,D]`，与 generated 同形状、浮点 dtype/device；N 为生成样本数，D 为特征维度，positive 的 M 个样本不会新增输出轴。每行是一个生成样本的场，不是 `[N,N+M]` 权重矩阵。不在此函数内 detach；冻结目标由 drifting_loss 负责。

#### `drifting_loss`（需实现）

返回形状 `()` 的浮点 Tensor，dtype/device 与 generated 一致。MSE 对 N 个生成样本和 D 个特征全部取均值。整个 generated+field 目标停止梯度，只有左侧 generated 接收梯度，positive 不接收；不修改输入。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `drift_field(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`
- `drifting_loss(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`

采用 exp(-欧氏距离/temperature)，屏蔽生成样本自身配对，以 sqrt(行质量×列质量) 归一化，再用正负组的交叉质量加权中心差。这是官方 toy 变体。整个 g+V 目标必须冻结，漏掉 detach 可能保持相同 loss 数值却让梯度归零。一步指生成器推理，不指只训练一步。

## 可运行实验

**来源：Official toy adaptation。** 改编作者 Swiss-roll/checkerboard notebook：从 noise32→3×256、batch2048、2000步缩为 noise8→2×48、生成96/正样本128、260步、temperature=.2，适合CPU判题。不含预训练特征、CFG或图像基准；论文 Figure 3 的双峰实验是另一个实验。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2602.04770
- https://github.com/lambertae/lambertae.github.io/blob/a7fa4baae52abbab2aaeb57136d557caee79ff39/projects/drifting/notebooks/drifting_model_demo.ipynb
- https://github.com/lambertae/drifting/tree/8861066430db11d2fb0858d042c7845b99df49f1

论文定位：arXiv v2 §3 drifting field and fixed-point loss；§5.1 Figure 3 bimodal toy；Official notebook compute_drift and drifting_loss。

数值改编：用 logsumexp 计算双向归一化核，替代 notebook 的直接 exp 与乘积截断。这样在原始指数下溢时仍保持数学归一化权重；有意移除了原代码绝对 1e-12 的乘积下限。空样本集及非正/非有限温度会报错。
