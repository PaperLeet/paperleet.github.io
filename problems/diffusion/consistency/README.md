# 21. Consistency Models：边界与教师目标

## 1. 题目目标

实现保持边界的 consistency 函数、冻结相邻时间目标的损失与 EMA 更新。

## 2. 接口与返回值

### 返回值要求

#### `consistency_function`（需实现）

返回 consistency 预测 Tensor，形状与 `noisy` 相同 `[B, ...]`；B 为样本数，省略号为每个样本的坐标维度。保留浮点 dtype/device 和模型梯度；`sigma=sigma_min` 的样本输出须等于原输入。不要 detach 整个预测。

#### `consistency_loss`（需实现）

返回形状 `()` 的浮点 Tensor 标量，dtype/device 与预测一致。对批次及全部坐标的平方误差求平均，不能返回逐样本向量或 Python float。保留 `prediction` 的梯度，在函数内部停止 `target_prediction` 的梯度。

#### `update_ema`（需实现）

返回 Python `None`。在 `torch.no_grad()` 下原地更新 `target` 的参数，保留各参数的形状、dtype/device；`online` 参数不变。此接口仅更新参数，不更新 buffers，也不返回新模型。

#### `teacher_euler`（已提供）

返回下降到 `next_sigma` 的一步教师样本 Tensor，形状与 `noisy` 相同 `[B, ...]`，保留浮点 dtype/device。B 为批大小，末尾为样本维度；已提供的 `no_grad` 装饰器使输出不带梯度。不原地修改输入，不返回轨迹。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `consistency_function(model, noisy: Tensor, sigma: Tensor, sigma_min: float = .002, sigma_data: float = .5) -> Tensor`
- `consistency_loss(prediction: Tensor, target_prediction: Tensor) -> Tensor`
- `update_ema(online, target, decay: float = .95) -> None`
- `teacher_euler(noisy: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor` (provided)

在 sigma_min 处函数必须严格等于输入，与网络输出无关。模型接收 c_in*x 与原始 sigma；本题提供原始 sigma 条件而非官方嵌入。skip/output 系数遵循官方边界参数化。较低噪声的 teacher/EMA 分支冻结。已提供 Euler helper 使用递减正噪声尺度。本题是带教师 consistency distillation，不是另外的无教师 consistency training。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 单点数据教师使概率流 Euler 步精确。小型 MLP 蒸馏验证边界和目标梯度路径，无需预训练模型。一半训练对使用正的相邻低噪声 sigma/2，EMA 网络真实参与目标；另一半连接严格 sigma_min 边界作为锚点。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2303.01469
- https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py

论文定位：arXiv v1 §3 consistency function boundary parameterization；§4 consistency distillation and EMA target；Official get_scalings_for_boundary_condition。
