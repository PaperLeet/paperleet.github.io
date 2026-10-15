# 17. JiT：高维空间中的干净数据预测

## 1. 题目目标

实现干净数据预测与速度空间监督，在高维嵌入数据上比较 x/epsilon/v 输出。

## 2. 接口与返回值

### 返回值要求

#### `make_noisy`（需实现）

返回带噪样本 Tensor，与 clean/noise 同形状 `[B,...]`、浮点 dtype/device；B 为批大小，末尾为样本坐标。t=0 为 noise，t=1 为 clean。保留可微输入路径，不返回时间或噪声的二元组。

#### `prediction_to_velocity`（需实现）

返回速度 Tensor，与 prediction/z 同形状 `[B,...]`、浮点 dtype/device，不做归约；B 为样本数，末尾为坐标维度。保留梯度；kind=`v` 时直接返回原 prediction（允许同一对象），其余类型返回转换后的速度。

#### `velocity_loss`（需实现）

返回形状 `()` 的浮点 Tensor，使用 prediction/样本的 dtype/device。所有参数化都在速度空间对 batch 及全部坐标的平方误差取均值，端点采用题目指定的截断目标。保留 prediction 的训练梯度；本函数没有主动 detach clean、noise、t 或目标。

#### 已提供的局部辅助函数

- `_time`：已提供：返回把 t 从 `[B]` reshape 成 `[B,1,...,1]` 的 Tensor，其阶数与 x 相同；B 为样本数，每个非 batch 轴长度为 1。保留 t（不是 x）的 dtype/device 与梯度，不修改输入；返回值可能与 t 共享存储。

### 公式与接口说明

### 接口

- `make_noisy(clean: Tensor, noise: Tensor, t: Tensor) -> Tensor`
- `prediction_to_velocity(prediction: Tensor, z: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`
- `velocity_loss(prediction: Tensor, clean: Tensor, noise: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`

远离截断端点时 v=x-epsilon；x 预测通过 (x_hat-z)/(1-t) 转换，epsilon 预测通过 (z-epsilon_hat)/t 转换。所有网络都使用速度损失。分母截断遵循官方 x 预测代码，参数化的精确等价性只在截断区域之外成立。t 为 [B]，应广播到其余维度。

## 可运行实验

**来源：Paper experiment reimplementation, reduced。** 根据 Figure 2 固定正交 2D→D 投影思想重实现，数据选用合成螺旋。默认 D=16、宽32、深3、160步；原文维度为2/8/16/512、宽256、深5。使用 toy.py 的 --impl 与 --full --steps 5000 可运行原维度/网络扫描；步数和螺旋数据为教学选择。短程结果不保证原论文排序，未找到原 toy 源码。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2511.13720
- https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py

论文定位：arXiv v1 §3.2 prediction and loss spaces；§3.3 Figure 2 high-dimensional manifold toy；Official Denoiser.forward。

实验 depth 计入输出层的全部 Linear 层，因此 --full 总计五个 Linear 层，之间使用 ReLU。
