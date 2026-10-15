# VICReg 方差、不变性与协方差

实现 [VICReg](https://arxiv.org/html/2105.04906v3) 的特征空间目标：同一样本的两个增强视图保持一致，每个特征在样本之间保留足够变化，不同特征减少冗余。

## 实现要求

### 返回值要求

#### `vicreg_terms` (需实现)

按顺序返回 `(invariance, variance, covariance)`，恰好三个标量浮点 `Tensor`，形状均为 `()`，保持两视图 dtype/device。对于 `[B,D]` 视图，invariance 平均 B*D 个平方差；variance 平均特征惩罚及两个视图；covariance 将两个视图的非对角平方项求和后除以 D。保留两视图及分母为 B-1 的样本统计量梯度，不 detach、不修改输入。

#### `vicreg_loss` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，按 `vicreg_terms` 声明的顺序对三个分量加权求和。保持共同输入 dtype/device 及 x、y 两侧梯度；不返回分量、Python float、detach 后的值，也不原地更新输入。

#### `_check_pair` (已提供)

已提供校验函数：有限且匹配的 float32/float64 视图 `[B,D]` 满足 B>=2、D>=1，epsilon 为有限正数时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_weights` (已提供)

已提供校验函数：三个系数均有限且非负时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

给定成对特征 $X,Y\in\mathbb R^{B\times D}$，令 $X_c=X-\operatorname{mean}_B(X)$，$C_X=X_c^TX_c/(B-1)$；对 $Y$ 同理。

$$
s=\operatorname{mean}((X-Y)^2),\qquad
v=\frac{1}{2D}\sum_j\left[\max(0,1-\sqrt{C_{X,jj}+\epsilon})+\max(0,1-\sqrt{C_{Y,jj}+\epsilon})\right],
$$

$$
c=\frac1D\sum_{i\ne j}(C_{X,ij}^2+C_{Y,ij}^2),\qquad
L=\lambda s+\mu v+\nu c.
$$

## 需要实现

| 函数 | 输出 |
|---|---|
| `vicreg_terms(x,y,eps=1e-4)` | 三个标量张量，顺序为不变性、方差、协方差 |
| `vicreg_loss(x,y,sim_coeff=25.,std_coeff=25.,cov_coeff=1.,eps=1e-4)` | 加权标量损失 |

两个输入具有相同的 float32/float64 类型、设备和 `(B,D)` 形状，`B>=2`、`D>=1`。输入检查已提供；保持输入内容、类型和设备，两个视图都要有梯度。系数是非负有限 Python 数值；epsilon 有限且大于零。

方差和协方差采用**样本统计**，分母为 `B-1`。不变性对全部 `B*D` 元素取平均；方差项对两个视图取平均；协方差项则对两个视图**求和**。非对角线平方和除以 `D`，不除以特征对数；`(i,j)` 和 `(j,i)` 都计入。`D=1` 时协方差项为零。

上述系数遵循已固定版本的官方代码。代码的 MSE 同时对样本和特征取平均，论文式 (5) 则写为向量距离平方对样本取平均。论文书面总目标对两视图的方差项求和，代码采用平均。本题统一使用代码中的二分之一因子和默认权重。

默认 epsilon 下，两个全零视图的三个损失为 `(0,0.99,0)`，但梯度仍为零。正的塌缩惩罚不代表优化一定能逃离精确塌缩；小幅度但**不恒定**的特征则不同。

## Toy：特征展开与去冗余

**教学改编。** 从小幅、相关但不恒定的 16 对二维特征出发，调用你的函数进行优化。图表展示损失，以及独立计算的标准差和非对角协方差；最后与精确塌缩的三个损失项比较。这里不训练图像编码器。

T0 检查接口；T1 用标量循环和单特征例子检查数值；T2 检查双视图梯度、精确塌缩驻点、平移与置换，以及重复特征的冗余；T3 使用你提交的损失优化特征几何。预览图由参考解生成。
