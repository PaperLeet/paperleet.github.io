# 19. DMD：分数差分布蒸馏

## 1. 题目目标

实现分布匹配生成器梯度、配对回归与独立的 fake-score 更新。

## 2. 接口与返回值

### 返回值要求

#### `score_difference`（需实现）

返回一个冻结的梯度目标 Tensor，与 `real_score` / `fake_score` 同形状 `[B,...]`、浮点 dtype/device；weight 的 B 个权重逐样本广播，不做归约。整个结果必须 detach，因此不向 real_score、fake_score 或 weight 传播梯度。

#### `dmd_loss`（需实现）

返回形状 `()` 的 Tensor 标量，使用 generated 的浮点 dtype/device。matching 与 paired regression 均对 B 和全部坐标取均值，matching 带 1/2 系数。只有左侧 generated 路径保留梯度；冻结由 generated-gradient 构造的目标及 paired_target。不返回梯度张量本身。

#### `fake_score_loss`（需实现）

返回形状 `()` 的浮点 Tensor，dtype/device 与 generated 一致；对批次及所有坐标的去噪 score 误差取均值。generated 在函数内部 detach，模型参数仍可训练；不要把整个损失 detach。本接口没有主动停止 noise 或 sigma 的梯度。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `score_difference(real_score: Tensor, fake_score: Tensor, weight: Tensor) -> Tensor`
- `dmd_loss(generated: Tensor, gradient: Tensor, paired_target: Tensor, regression_weight: float = .1) -> Tensor`
- `fake_score_loss(model, generated: Tensor, noise: Tensor, sigma: Tensor) -> Tensor`

采用 VE 加噪 y=x+sigma*noise。输入是 ∇y log q_sigma 与 ∇y log p_sigma，不是 epsilon 预测。正权重由调用者提供。匹配替代损失的梯度为 weight*(s_fake-s_real)/numel，分数目标冻结。DMD 保留配对回归；高斯 toy 用 MSE 替代图像回归度量。训练 fake-score 的 DSM 不能更新生成器。链接 DMD2 仓库仅用于共同梯度机制，不将本题混为 DMD2。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 平移二维高斯生成器和可训练高斯 fake score 展示分数符号、梯度隔离与配对回归。这不是论文 Figure 3 双峰原代码，而是用解析分数隔离相同分布匹配机制。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2311.18828
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

论文定位：arXiv v1 §3 distribution matching gradient and regression regularization；Figure 3 distribution matching illustration。

第二个 T3 与参考预览增加 **Figure 3 启发的双峰改编实验**：解析双高斯学生的两个中心都从右侧附近出发，比较仅真实分数、分数差匹配、匹配加配对 MSE 回归。这里教师和 fake score 都使用解析混合分数以隔离生成器梯度；第一个高斯 T3 单独验证 fake score 训练。CPU 运行600步，噪声 sigma 从1.5降至0.6，目标中心为 (-1.8,0)/(1.8,0)，分量标准差 .25，报告每个模式占比和覆盖数。这些配置与曲线是 PaperLeet 重实现，不是原 Figure 3 数据或源码。
