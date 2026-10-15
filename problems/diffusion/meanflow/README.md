# 16. MeanFlow：平均速度与 JVP 目标

## 1. 题目目标

实现平均速度的全导数目标、冻结目标损失，以及一次前向采样。

## 2. 接口与返回值

### 返回值要求

#### `meanflow_target`（需实现）

按顺序返回 `(prediction, target)` 二元组，两项均为与 z 同形状 `[B,...]`、浮点 dtype/device 的 Tensor；B 为样本数，末尾为坐标维度。prediction 是模型原预测并保留训练梯度；target 是完全 detach 的 JVP 修正目标。不返回 JVP 导数本身。

#### `meanflow_loss`（需实现）

返回形状 `()` 的浮点 Tensor 标量，dtype/device 与 clean/noise 和模型输出一致。对 batch 及全部坐标的 prediction-target 平方误差取均值，无自适应权重。通过 prediction 保留模型梯度，target 停止梯度。

#### `one_step`（需实现）

返回一次模型调用得到的数据端 Tensor，与 noise 同形状 `[B,...]`、浮点 dtype/device；B 为样本数，末尾为坐标。只返回样本，不返回轨迹；已提供的 no_grad 装饰器使输出无梯度，不修改 noise。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]`
- `meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor`
- `one_step(model, noise: Tensor) -> Tensor`

接口使用论文顺序 `(z,r,t)`；JVP 方向 `(v,0,1)` 固定 r。t 增大表示噪声增多。目标为 `stopgrad(v-(t-r)du/dt)`。本题用逐元素平均 MSE，不包含官方图像实现的自适应权重。r=t 时目标退化为条件 Flow Matching，并不要求未训练网络立即等于 v。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 使用解析单点数据族与学员 JVP 目标学习中心；这是机制演示，不声称是原论文低维实验。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2505.13447
- https://github.com/Gsunshine/py-meanflow/blob/1f6d72d94247c8fdeb89489acca1a8007a6baf6c/meanflow/models/meanflow.py

论文定位：arXiv v1 §3 MeanFlow identity and stop-gradient training objective；Official forward_with_loss and sample。
