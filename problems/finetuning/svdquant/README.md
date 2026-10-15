# SVDQuant 平滑、低秩与残差量化

实现 [SVDQuant v1 §4.2 Eq. (5)](https://arxiv.org/html/2411.05007v1) 的数值核心：把激活离群值迁移到权重，保留高精度主导低秩分量，对残差分支做量化。

## 函数

### 返回值要求

#### `smooth_inputs` — 需要实现

返回浮点 Tensor 的 Python 元组 `(xs, ws)`，形状为 `[B,M]`、`[M,N]`，顺序对应 x、weight。B 为批大小，M 为输入通道数，N 为输出通道数；平滑后的两项满足 xs@ws=x@weight。保持共同 dtype/device，不修改输入。本题要求推理数值结果，不要求量化流水线的梯度。

#### `split_weight` — 需要实现

依次返回浮点 Tensor 元组 `(left, right, residual)`，形状为 `[M,r]`、`[r,N]`、`[M,N]`；weight 为 `[M,N]`，r=rank。left@right 是最佳 rank-r 近似，residual=weight-left@right。接受等价的因子符号/基，不以奇异值代替 residual。rank=0 时因子对应轴为空、残差为完整 weight；满秩时残差数值上为零。保持 dtype/device，不修改 weight；SVD 梯度不在范围内。

#### `svdquant_linear` — 需要实现

返回一个浮点 Tensor `[B,N]`；x 为 `[B,M]`、weight 为 `[M,N]`，B 为批大小，M/N 为输入/输出通道数。它是高精度低秩分支与 fake-quantized 残差分支相加后的输出，不返回分支元组。保持共同 dtype/device 和输入值；满秩时在数值误差内恢复 x@weight。本题为推理接口，不要求 SVD/舍入的梯度。

#### `fake_quantize` — 已提供

返回一个与 x 同形状/dtype/device 的浮点 Tensor，表示逐 Tensor 对称 INT 网格量化再反量化后的值，不是整数编码或 `(codes,scale)` 元组。全零输入返回零，不修改输入；此已提供舍入 helper 的梯度不属于本题要求。

#### `_check_smoothing` — 已提供 helper

校验非空 x[B,M]、weight[M,N] 和有限正值 scale[M] 及匹配的输入通道数后返回 None。受检布局/尺度不合法时抛出 ValueError；不修改或返回 Tensor。

#### `_check_rank` — 已提供 helper

weight 为矩阵且 rank 为 [0,min(weight.shape)] 内的 Python int 时返回 None。秩/布局不合法时抛出 ValueError，不返回分解结果，不修改输入。

### 公式与约定

采用 `x[B,M]`、`weight[M,N]`、正且有限的 `scale[M]`。未量化输出是 `x@weight`，与 PyTorch Linear 的转置权重布局不同。

1. `smooth_inputs(x,weight,scale)` 返回 `(xs,ws)`：`xs=x/scale`、`ws=scale[:,None]*weight`，应保持 `xs@ws=x@weight`。
2. `split_weight(weight,rank)` 返回 `(left[M,r],right[r,N],residual[M,N])`，`left@right` 为截断 SVD 的最佳 rank-r 近似，`residual=weight-left@right`。允许奇异向量符号及等价因子分解不同，覆盖 `rank=0` 和 `rank=min(M,N)`。
3. `svdquant_linear(x,weight,scale,rank,bits=4)`：先平滑，再分解**平滑后的权重**，计算：

$$Y=(\hat X L_1)L_2+Q(\hat X)Q(R).$$

低秩分支必须接收**未量化的平滑激活**；低比特分支量化的是残差 R，不能量化完整平滑权重。已提供 `fake_quantize`：每个张量共享 `s=max(abs(z))/(2^(bits-1)-1)`，`Q(z)=s*clamp(round(z/s),-qmax,qmax)`，中点取偶数，零张量输出零，`2<=bits<=8`。

starter 已提供校验和量化。输入为有限 CPU float32/float64，同 dtype/device，维度非空。保留 dtype/device 与输入值，兼容非连续输入。本题为推理数值接口，不考查 SVD/rounding 梯度。

## Toy 实验

T3 构造含一个 30 倍通道的合成激活，通过提交函数计算输出误差及奇异值曲线，比较直接 W4A4、仅平滑与 rank 0–6 双分支结果。这是 Figures 4–5 的**教学改编**：平滑系数已给定，只做一次 SVD；高精度分支使用输入浮点 dtype，不模拟硬件 FP16。不含模型权重、校准搜索、迭代细化、FP4 编码、整数内核或 Nunchaku 延迟复现。核心同时包括平滑、低秩、残差量化，不将单独 SVD 当作完整系统。
