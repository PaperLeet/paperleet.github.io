# GroupNorm：分组统计与批次独立性

## 1. 这道题做什么

### 返回值要求

#### `group_norm` · 需要实现

返回一个 Tensor，形状与输入完全相同，均为 `[N,C,*spatial]`。保留输入的 dtype/device，以及对 `x`、`weight`、`bias` 的梯度；不修改输入。每个样本内按连续通道分组计算统计量，再使用逐通道的缩放和偏置。

### 公式与约定

实现 [Group Normalization](https://arxiv.org/html/1803.08494v3#S3) 的核心函数。论文由 Yuxin Wu、Kaiming He 提出，发表于 ECCV 2018。式 (7) 定义统计范围，图 3 展示实现结构。先根据公式推导需要的张量轴，再阅读参考解。

## 2. 核心公式

同一样本、同一通道组的 $m$ 个通道与空间位置共享统计量：

$$
\mu=\frac1m\sum_jx_j,\qquad
v=\frac1m\sum_j(x_j-\mu)^2,\qquad
y_i=\gamma_{c(i)}\frac{x_i-\mu}{\sqrt{v+\varepsilon}}+\beta_{c(i)}.
$$

方差分母为 $m$，不是 $m-1$。均值与方差按组共享，仿射参数仍然按通道独立；不能跨 batch 维计算均值或方差。[PyTorch 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)同样采用总体方差和逐通道仿射参数。

## 3. 从公式到代码

把通道轴拆成 `[G,C/G]`，保留样本轴与组轴，在组内通道和全部空间轴上求统计量。恢复输入形状后，将 `[C]` 的 `weight` 和 `bias` 广播到通道轴。

例如四个通道依次为 `[0,2]`、`[4,6]`、`[8,10]`、`[12,14]`，`G=2` 时两组均值为 3 和 11，总体方差均为 5。同一组的通道仍可使用不同的仿射参数。

## 4. 接口约定

| 项目 | 要求 |
| --- | --- |
| 输入 | `[N,C,*spatial]`，至少三维；`N>=0`，通道数和空间大小为正 |
| 仿射参数 | `weight [C]`、`bias [C]`，dtype/device 与输入一致 |
| 组数 | 整数且不是 bool，`1<=G<=C`、`C%G==0`；非法值抛出 `ValueError` |
| 精度 | float32/float64；测试采用尺度适中的有限输入和有限正 epsilon |
| 统计量 | 每个样本内的连续通道组与空间位置；总体方差，epsilon 在根号内 |
| 梯度 | 均值与方差必须保留计算图，不能 detach |

本题是无状态函数，将论文图像布局推广到一个或多个空间轴。空 batch 保留形状。单元素组按数学公式处理：归一化值与输入/缩放梯度为零，输出等于对应通道的偏置；该边界直接用公式验证，不依赖库函数的输入大小限制。

## 5. 看实验

固定一张 anchor 特征图，仅改变同批其他样本的数量、尺度和偏移。第一张图展示 anchor 输出的最大变化：你的 GN 应保持不变，提供的训练态 batch 统计基线则会变化。第二张图比较同一样本在 `G=1`、`G=2`、`G=C` 时的输出。

**来源说明：** 这是小规模合成教学实验，用于验证归一化轴与样本独立性，没有训练 ImageNet/COCO 模型，也不复现论文准确率或硬件速度。你的运行图表调用你提交的函数，参考预览使用参考实现。

## 6. 怎么运行

在网页中补全代码并运行全套测试，或使用本地命令：

```bash
paperleet test architecture/group_norm --impl problems/architecture/group_norm/solution.py
paperleet mutate architecture/group_norm
```

## 7. 继续思考

T0 检查接口、精度、非连续输入与输入保留；T1 使用手算结果；T2 用标量与库函数基准、解析梯度、有限差分及边界条件检查实现；T3 测量实际函数的批次组成响应。

解释 `G=1` 的全层统计量和 `G=C` 的逐实例统计量。存在空间维时，全层统计范围包括通道与空间，不要与 Transformer 逐 token 的 LayerNorm 混淆。再推导同组不同通道之间的梯度耦合。
