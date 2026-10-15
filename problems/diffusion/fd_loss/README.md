# FD-loss 可微 Fréchet 损失

## 1. 题目目标

实现 [Representation Fréchet Loss for Visual Generation](https://arxiv.org/html/2604.28190) 的训练目标，对生成总体统计求梯度，并冻结真实参考统计。

## 2. 接口与返回值

### 返回值要求

#### `frechet_distance`（需实现）

返回形状 `()` 的 Tensor 标量 FD，保持输入 float32/float64 dtype/device。D 维均值与 D×D 协方差归约成一个距离，不除以 D。对生成 mean/covariance 保留梯度，对 reference_mean/reference_covariance 停止梯度；梯度保证要求两者协方差均为正定。半正定仅保证前向值，最终 FD 不强制裁剪为非负，不修改输入。

#### `normalized_fd_loss`（需实现）

返回形状 `()` 的 Tensor 标量，与 distances 同 dtype/device。对 R 个表征的加权归一化距离求和，不除以 R 或权重和。只让每项分子 distances 保留梯度，分母中的 distances 和 weights 均 detach；不修改输入。

#### 已提供的局部辅助函数

- `_check_gaussian`：已提供：当均值和协方差满足高斯统计约定时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。
- `_check_pair`：已提供：当生成与参考统计的形状、dtype/device 匹配时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。
- `_check_normalization`：已提供：当距离、非负权重和正分母合法时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。

### 公式与接口说明

特征高斯分布的 Eq.(2) 为

$$
FD=\|\mu_g-\mu_r\|_2^2+\operatorname{tr}(\Sigma_g)+\operatorname{tr}(\Sigma_r)
-2\operatorname{tr}\left[(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\right].
$$

对称夹积的平方根迹与原始协方差乘积公式一致。附录 B.3 用 `torch.linalg.eigvalsh` 求它的特征值，避免通过生成统计的特征向量反向传播。两个协方差通常**不可交换**。

对于每个表征的距离 `d_i`，Eq.(6) 使用

$$
\mathcal L=\sum_i w_i\frac{d_i}{\operatorname{sg}(d_i)+\epsilon}.
$$

### 需要实现

| 函数 | 输入 | 输出 |
|---|---|---|
| `frechet_distance(mean, covariance, reference_mean, reference_covariance)` | `(D,)` 均值与 `(D,D)` 协方差 | 标量 FD |
| `normalized_fd_loss(distances, weights, eps=.01)` | 相同形状的非空 `(R,)` 向量与有限正 epsilon | 加权归一化距离之**和** |

所有张量的类型和设备相同，使用 float32 或 float64。参数检查已经提供，保留类型和输入数值。在函数内部对真实均值、真实协方差和表征权重停止梯度。生成均值、生成协方差以及距离分子必须保留梯度。

通过**已 detach 的**参考协方差的特征分解构造参考平方根。计算夹积，使用 `(A+A.T)/2` 对称化，再把其特征值开方求和。开方前将数值误差造成的负特征值夹到零。不要添加对角 jitter、逐元素开方、裁剪最终 FD，或对非对称的协方差乘积直接调用 `eigvalsh`。

前向测试支持半正定协方差，包括零特征值。**梯度保证要求生成与参考两个协方差均为正定**，其中也包含重复正特征值的情况。零特征值处的平方根导数不在本题承诺范围内，不用隐式正则化改变目标。

一维示例：均值为 `1` 和 `0`，方差为 `4` 和 `1`，FD 为 `1+(2-1)^2=2`。将它用权重 `1`、epsilon `.01` 归一化，值为 `2/2.01`，但对这个距离的导数是 **`1/2.01`**，因为分母停止梯度。允许 FD 的微小负舍入误差，只要求 `distance+eps` 仍为正；函数不会偷偷裁剪距离。

官方默认使用相同权重和 `.01` 的 epsilon。这个接口也接受其他固定非负权重与有限正 epsilon，不额外除以表征数或权重和。多个冻结特征模型、真实统计和图像编码器属于已提供的上下文，不是本题需要训练的函数。

## Toy：二维生成器后训练

**来源：教学改编。** 一个五参数仿射生成器变换四个对称噪声点。脚手架提供 EMA 总体矩（`beta=.8`），你的 FD 和归一化函数使生成器经过 200 步训练拟合一个相关高斯分布。本实验只用 CPU 张量，采用 `.1` 的 epsilon 和小型 Adam 优化器，不复现 ImageNet 生成器或论文生产超参数。

图表分开展示原始 EMA FD、独立二维公式计算的当前生成器 FD，以及归一化训练目标。归一化值接近一时，原始距离仍可能明显变化，因此用**原始 FD**判断分布拟合进展，不要只看接近一的目标值。

T0 检查接口；T1 使用对角和不可交换二维协方差的闭式答案；T2 检查梯度、冻结参考、重复正谱、停止归一化分母梯度、对称性与坐标变换；T3 真正调用提交函数训练生成器。预览图使用参考实现。
