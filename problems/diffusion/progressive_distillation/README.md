# 26. Progressive Distillation：教师两步压成学生一步

## 1. 题目目标

求出使学生一步 DDIM 精确匹配教师两步的干净数据目标，并训练学生。

## 2. 接口与返回值

### 返回值要求

#### `ddim_step`（需实现）

返回较干净时间 s 的样本 Tensor，与 noisy 同形状 `[B,...]`、浮点 dtype/device；B 为批大小，末尾为样本坐标。只返回更新后的样本，不返回 clean_prediction 或系数。保留可微路径；时间系数 `[B]` 在样本维度广播。

#### `distilled_target`（需实现）

返回使学生一步匹配教师终点的干净预测目标 Tensor，与 noisy 同形状 `[B,...]`、浮点 dtype/device。整个结果必须 detach，不向 noisy、teacher_endpoint 或时间系数传播梯度。不是教师终点本身，也不是教师第一次预测。

#### `distillation_loss`（需实现）

返回形状 `()` 的浮点 Tensor，dtype/device 与 prediction 一致。先将每个样本的 weight 广播到坐标，再对 batch 与所有坐标的加权平方误差取均值；不除以权重和。保留 prediction 梯度并在函数内部 detach target；weight 不主动 detach。

#### `teacher_pair`（已提供）

返回教师恰好两步 DDIM 后的终点 Tensor，与 noisy 同形状 `[B,...]`、浮点 dtype/device。只返回最终 end 时间的样本，不返回中点或两个预测的元组；输出不带梯度，不修改 noisy。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `ddim_step(noisy: Tensor, clean_prediction: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distilled_target(noisy: Tensor, teacher_endpoint: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distillation_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor`
- `teacher_pair(teacher, noisy: Tensor, t: Tensor, midpoint: Tensor, end: Tensor) -> Tensor` (provided)

alpha/sigma 是幅度系数，alpha²+sigma²=1，不是累计方差 alpha_bar。要求 sigma_t>0 且结束时间严格更小，保证分母非零。教师做两次去噪，在中间状态重新预测。解出的学生目标必须冻结。每样本权重由调用者给出，toy 使用1；参数化旋转仍在第7题。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 固定仿射去噪教师在余弦调度下走两步 DDIM，小型线性学生学习解出的目标，一次前向匹配终点。演示精确蒸馏代数，不代表图像质量或多阶段反复减半。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2202.00512
- https://github.com/google-research/google-research/tree/0a1b0fe7298a58bca6dcb47112379d829bc0c0a5/diffusion_distillation

论文定位：arXiv v1 §3 Progressive distillation；Algorithm 2 and Appendix B distillation target。
