# LeJEPA 切片高斯正则 SIGReg

实现 [LeJEPA 第 4.2.3、5.1 节](https://arxiv.org/html/2511.08544v1) 的 SIGReg：把特征投影到单位方向，比较各切片的经验特征函数与标准高斯目标，再对切片统计量取平均。这使约束超出一阶和二阶矩。

## 实现要求

### 返回值要求

#### `epps_pulley` (需实现)

返回一个浮点 `Tensor` `[M]`，对应 `projected[N,M]` 的 M 个投影，每项是包含 N 倍系数的 Epps-Pulley 统计量。积分归约频率节点轴，此处不平均 M 个切片。保持 dtype/device 与投影样本梯度，冻结节点 t，不修改输入。

#### `sigreg_loss` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，为特征 `[N,D]` 沿方向 `[D,M]` 得到的 M 个切片统计量的平均值。保持特征 dtype/device 和梯度；方向与频率节点 t 都是冻结常量。不要额外除以 N、detach 特征或修改输入。

#### `_check_projected` (已提供)

已提供校验函数：projected `[N,M]` 是有限、非空的 float32/float64 Tensor，匹配的节点 t `[K]`（K>=2）从零开始并严格递增时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_features` (已提供)

已提供校验函数：features `[N,D]` 与节点通过校验，有限且匹配的 directions `[D,M]` 每列为单位长度时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

对投影 $P=XA\in\mathbb R^{N\times M}$ 与频率 $t_k\ge0$，

$$
\hat c_{mk}=\frac1N\sum_n\cos(t_kP_{nm}),\quad
\hat s_{mk}=\frac1N\sum_n\sin(t_kP_{nm}),\quad
\phi_k=e^{-t_k^2/2},
$$

$$
E_{mk}=\left[(\hat c_{mk}-\phi_k)^2+\hat s_{mk}^2\right]\phi_k,\quad
T_m=2N\sum_{k=0}^{K-2}\frac{t_{k+1}-t_k}{2}(E_{m,k+1}+E_{mk}),\quad
L=\operatorname{mean}_m T_m.
$$

## 需要实现

| 函数 | 输出 |
|---|---|
| `epps_pulley(projected,t)` | 每个切片一个统计量，形状 `(M,)` |
| `sigreg_loss(features,directions,t)` | 对切片取平均后的标量 |

`features` 形状为 `(N,D)`，`directions` 为 `(D,M)` 且每列是单位向量。`t` 从零开始严格递增，至少包含两个节点；`N,D,M>=1`。所有输入具有相同的 float32/float64 类型和设备。输入检查已提供，保持输入内容、类型与设备。

按真实网格间隔积分，支持非均匀网格。通常提供 `[0,3]` 上的 17 个节点；正半轴积分乘二包含负半轴贡献。高斯因子同时作为目标特征函数和积分窗口。保留正弦项，乘以样本数 `N`，最后对切片取平均。

方向和积分节点属于固定支架，在函数内 detach；特征与投影样本必须保留梯度。不要中心化、标准化、白化或 detach 特征，目标明确为 `N(0,I)`。本题直接提供方向，不要求随机抽样，以便得到确定的判题结果。

全零特征具有正统计量，但**梯度为零**；附近不恒定的特征可能存在下降方向。有限方向和有限积分节点不能保证识别任意分布。

## Toy：相同均值方差，不同分布

**教学改编。** 用 16 个固定切片优化 64 个变形的二维特征；同时将高斯分位数样本和均衡的 `{-1,+1}` 样本调整为精确零均值、单位总体方差，比较它们仍然不同的特征函数。图表展示你提交的损失、独立的特征统计，以及一个投影的特征函数。不训练编码器或完整的 LeJEPA 多视图目标。

T1 使用完整对称网格上的标量复数运算作为 oracle；T2 检查梯度、固定支架、塌缩驻点、联合旋转、样本数缩放以及目标均值和尺度；T3 在 toy 中运行你提交的函数，并用独立的完整网格 oracle 核对最终损失。预览图由参考解生成。
