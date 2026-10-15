# MoGe 鲁棒最优对齐

实现 [MoGe](https://arxiv.org/abs/2410.19115) 的鲁棒最优对齐（ROE）核心。预测点图存在未知全局尺度，以及相机 z 轴方向的平移。目标使用真实深度倒数作权重，逐坐标计算绝对残差：

## 实现要求

### 返回值要求

#### `optimal_scale_1d` (需实现)

按顺序返回 Python tuple `(scale, loss)`；两项都是已 detach 的标量浮点 `Tensor`，形状 `()`，dtype/device 与 x 一致。`loss` 是加权绝对残差之和，指定 `trunc` 时逐项截断，不取平均。并列最优时选最小的有效 scale；没有有效系数时返回零 scale 及其目标值。不修改输入。

#### `roe_align` (需实现)

按顺序返回 Python tuple `(scale, z_shift)`，两项都是已 detach 的标量浮点 `Tensor`，形状 `()`，保持点图 dtype/device。`z_shift` 只表示 z 轴平移，不是三维向量，第二项也不是 loss。只拟合 mask 选中的有效点；不修改输入，本题不要求求解器梯度。

#### `_check_vectors` (已提供)

已提供校验函数：x、y、非负 weight 是有限、非空且匹配的一维浮点 Tensor，trunc 为 None 或有限正数时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_points` (已提供)

已提供 helper：返回 `(valid_prediction, valid_target)`，两个浮点 Tensor `[M,3]` 按 mask 选中点的原顺序排列，M 为 True 数量。保持点图 dtype/device，不修改原输入。校验非空选择与有效目标的正深度，被校验条件不满足时抛出 `ValueError`；未选中的 NaN 在数值检查前被排除。helper 自身不 detach，求解器调用方使用 no_grad。

### 公式与约定

$$
\min_{s,t}\sum_{i\in M}\sum_{d\in\{x,y,z\}}
\min\left(\frac{|s\hat p_{id}+t\mathbf 1_{d=z}-p_{id}|}{p_{iz}},\tau\right).
$$

`trunc=None` 时不截断。必须先乘权重，再对每个坐标的残差截断。旋转及 x/y 平移固定不变；本题优化的尺度是无约束实数。

## 实现接口

| 函数 | 输入 | 输出 |
|---|---|---|
| `optimal_scale_1d(x, y, weight, trunc=None)` | 三个同形状非空一维浮点张量，权重非负 | 标量 `scale`、最小目标值 |
| `roe_align(prediction, target, mask, trunc=None)` | 两个 `(N,3)` 点图及 `(N,)` 布尔掩码 | 标量 `scale`、标量 `z_shift` |

第一个函数求解 `sum(min(weight * abs(scale*x-y), trunc))`。全局最优解可在有效零残差折点 `y[i]/x[i]` 中找到，其中 `x[i] != 0` 且 `weight[i] > 0`。截断会增加折点，但这些折点的斜率变化不能创造严格局部极小值。多个折点目标值相同时选择最小尺度；没有有效系数时返回尺度零。

对点图，枚举有效锚点 `k`，令 `t = target[k,2] - s*prediction[k,2]`。只把所有点的 z 坐标减去锚点深度，保留 x/y 原值，每个锚点就得到一个一维尺度问题。选择目标值最小的候选。完整对齐允许返回任意全局最优解。

starter 已提供验证：有效目标深度必须为正、有限，输入使用相同 dtype/device 的 CPU float32/float64。先选取掩码中的有效点再计算；无效点允许含 NaN。不得修改输入，输出为相同 dtype/device 的、已 detach 的标量张量。作者训练代码会从选中锚点重建可微参数，这部分梯度路径不在本题范围内。

题目使用小规模精确枚举（测试 `N <= 64`），无需网络预测、多尺度球域或相机恢复。

## Toy：恢复带离群点的点云

**来源分类：teaching adaptation。** 六个生成点使用已知尺度与 z 平移，其中一个高杠杆坐标被破坏。你的函数绘制尺度目标曲线和恢复后的深度。截断 ROE 恢复内点，未截断拟合受到离群点影响。这是教学实验，不是 MoGe 原生 benchmark。

T0 检查接口；T1 使用独立标量计算与二维直线排列交点 oracle；T2 检查权重、截断顺序、掩码、平局与单位缩放；T3 调用提交函数产生图表。预览使用参考实现。

算法对照[作者的 `align` 与 `align_points_scale_z_shift`](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/alignment.py)。教学版直接计算候选值，生产代码使用排序和 scatter 加速。
