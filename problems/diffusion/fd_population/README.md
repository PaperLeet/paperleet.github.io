# FD-loss 总体矩与 EMA

## 1. 题目目标

实现 [Representation Fréchet Loss for Visual Generation 第 3.2 节](https://arxiv.org/html/2604.28190)中的总体统计估计器。FD-loss 用历史信息估计特征分布，同时只对当前微批次求梯度。论文默认的 EMA 版本保存均值和**原始二阶矩**。

## 2. 接口与返回值

### 返回值要求

#### `batch_moments`（需实现）

按顺序返回 `(mean, raw_second)` 二元组，均为 Tensor。输入 features 为 `[B,D]`，B 为样本数、D 为特征数；mean 为 `[D]`，raw_second 为 `[D,D]` 的原始二阶矩，不是中心化协方差。两项均除以 B 而非 B-1，保留 features 的 float32/float64 dtype/device 和梯度，不修改输入。

#### `ema_population_stats`（需实现）

按顺序返回三元组 `(mean, raw_second, covariance)`，Tensor 形状依次为 `[D]`、`[D,D]`、`[D,D]`；D 为特征数。保留 features 的 float32/float64 dtype/device，以及当前批次梯度（beta=1 时为零）；在函数内 detach 两个历史张量。不原地更新历史，调用方另行保存 detach 后的 mean/raw_second。

#### 已提供的局部辅助函数

- `_check_features`：已提供：当特征是合法有限 `[B,D]` 浮点张量时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。
- `_check_history`：已提供：当当前特征、历史矩和 beta 匹配约定时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。

### 公式与接口说明

对当前特征 $X\in\mathbb R^{B\times D}$，先计算

$$
\mu_b=\frac1B\sum_i x_i,\qquad M_b=\frac1B\sum_i x_i x_i^\top.
$$

再实现 Eq.(4)-(5)：

$$
\mu=\beta\,\operatorname{sg}(\mu_{old})+(1-\beta)\mu_b,\quad
M=\beta\,\operatorname{sg}(M_{old})+(1-\beta)M_b,\quad
\Sigma=M-\mu\mu^\top.
$$

### 需要实现

| 函数 | 返回值 |
|---|---|
| `batch_moments(features)` | `(mean, raw_second)`，形状分别为 `(D,)`、`(D,D)` |
| `ema_population_stats(features, old_mean, old_second, beta)` | 按顺序返回 `(mean, raw_second, covariance)` |

输入使用相同的 float32/float64 类型和设备，`B>=1`、`D>=1`，`beta` 在 `[0,1]` 内；端点用于测试。历史输入表示有效且有限的总体矩。类型、维度和有限值检查已提供。保留输入数值，不要原地更新历史。

所有批次矩都除以 **B**，`B=1` 时也成立。`raw_second` 是 `E[xx^T]`，不是中心化协方差。在 `ema_population_stats` 内部对两个历史张量停止梯度，即使调用方给它们设置了 `requires_grad=True`。返回的三个统计量必须保留当前批次的梯度。调用方使用 `mean.detach()`、`raw_second.detach()` 保存下一步历史，脚手架负责这部分状态管理。

例如：当前单个特征为 `[3]`，历史均值为 `[1]`、原始二阶矩为 `[[2]]`，`beta=.5`。应得到均值 `[2]`、原始二阶矩 `[[5.5]]`、协方差 `[[1.5]]`。直接对两个协方差做 EMA 会错误地得到 `[[.5]]`。

本题使用论文的 **EMA 总体矩约定**。官方 FIFO 队列分支使用 `N-1` 计算样本协方差，那是另一种估计器，不属于本题范围；不要把它的 Bessel 修正加进 EMA 公式。

## Toy：小批次中的总体信息

**来源：教学改编。** 两组平移后的二维特征总共包含八个点。四十个微批次交替来自两组，因此单个批次不能代表总体。脚手架用这八个点暖启动 EMA，再调用你的函数比较当前批次与 EMA 的估计误差。图表展示均值、方差和总体矩误差。这是为理解估计器构造的确定性特征流，不是论文原图，也不代表变化中的真实数据总会得到同样结果。

原论文在后训练之前用基础生成器暖启动统计量。本实验直接提供特征以隔离统计估计器。Fréchet 损失、图像编码器、多设备收集与生成器优化不在本题范围内。

T0 检查接口；T1 使用标量矩和显式加权总体作为独立答案；T2 验证停止历史梯度、当前梯度解析式、EMA 端点、仿射变换和多步几何权重；T3 用你的提交运行特征流。预览图使用参考实现。
