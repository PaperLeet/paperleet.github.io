# 20. DMD2：双时间尺度与对抗更新

## 1. 题目目标

在分布匹配基础上实现 critic/生成器更新比例和非饱和 GAN 项。

## 2. 接口与返回值

### 返回值要求

#### `generator_due`（需实现）

返回 Python `bool`，不是整数或 Tensor。仅在已完成的 critic 更新数为正、且恰好达到一次 generator 更新周期时返回 True；计数为 0 时返回 False。此函数不执行更新，也没有张量形状或梯度。

#### `adversarial_losses`（需实现）

按顺序返回 `(discriminator_loss, generator_loss)` 二元组，每项均为形状 `()` 的浮点 Tensor。保留输入 logits 的 dtype/device；real 与 fake 数量可不同。判别器的 real/fake 项分别求均值再相加，生成器项对 fake 求均值。这里不 detach logits，两项保留各自的梯度路径。

#### `generator_loss`（需实现）

返回形状 `()` 的浮点 Tensor，使用 generated/logits 的 dtype/device。DM matching 对 generated 的全部元素取均值并带 1/2 系数，GAN 项对 fake_logits 取均值。冻结由 generated-gradient 构造的目标；保留 generated 与 fake_logits 的训练梯度。没有 paired regression 项，不返回多个损失。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool`
- `adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]`
- `generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor`

本题 critic/fake diffusion 每更新5次，生成器更新1次；计数表示已完成的 critic 更新，第0次不触发。判别器与生成器在 fake logits 上符号相反。调用方在 critic 训练时 detach 生成数据，在生成器训练时冻结 critic 参数但保留输入梯度。与 DMD 不同，生成器不含配对教师回归。本题聚焦稳定训练和真实数据贡献，不覆盖多步 backward simulation。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 非配对平移高斯 toy 将 fake-score 和判别器训练5次再更新学生1次，展示调度、GAN梯度方向与移除配对回归。用高斯分数参数替代预训练扩散特征。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2405.14867
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

论文定位：arXiv v1 §3 two-time-scale updates and GAN loss；Official EDM compute_generator_clean_cls_loss and compute_guidance_clean_cls_loss。
