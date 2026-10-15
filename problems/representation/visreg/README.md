# VISReg 尺度与分布形状正则

实现 [VISReg](https://arxiv.org/abs/2606.02572) 第 3.1 节的正则项与视图一致性目标。特征方差合适，不代表分布形状合适；这里分别优化这两个性质。

## 实现内容

### 返回值要求

#### `visreg_terms` (需实现)

按顺序返回 `(scale, shape, center)`，恰好三个标量浮点 `Tensor`，形状均为 `()`，dtype/device 与 z 一致。对 z `[N,D]` 和 K 个方向，scale、center 在 D 个特征上平均，shape 在 N*K 个投影分位误差上平均。保留 z 梯度，冻结方向，仅在 shape 分母中冻结标准差；scale 仍保留标准差梯度。不修改输入。

#### `view_invariance` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，保持 views 的 dtype/device：相对于前 G=`num_global` 个视图均值的平方误差，在 views `[V,N,D]` 的全部 V*N*D 个元素上平均。所有视图及全局视图均值都保留梯度，它不是 detach 的 teacher。不修改输入。

#### `_features` (已提供)

已提供校验函数：x 是指定秩的有限、非空 float32/float64 Tensor时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_terms` (已提供)

已提供校验函数：特征 z `[N,D]` 满足 N>=2，匹配的有限 directions `[D,K]` 每列非零，epsilon 为有限正数时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

`visreg_terms(z, directions, eps=1e-4)` 接收 `[N,D]` 特征和固定的非零投影列 `[D,K]`，依次返回三个标量：**scale、shape、center**。

每个特征维度的总体均值、标准差为

$$\mu_j=\frac1N\sum_n z_{nj},\qquad \sigma_j=\sqrt{\frac1N\sum_n(z_{nj}-\mu_j)^2}.$$

计算

$$L_{scale}=\operatorname{mean}_j(\sigma_j-1)^2,\quad
\widetilde z=\frac{z-\mu}{\operatorname{sg}(\sigma)+\epsilon},\quad
L_{center}=\operatorname{mean}_j\mu_j^2.$$

先把每个投影列归一化为单位向量，再投影 `z_tilde`，沿 **N 个样本的维度，对每个方向分别排序**。第 `i=1,...,N` 个标准高斯分位点为 `q_i=Phi_inverse(i/(N+1))`：

$$L_{shape}=\frac1{NK}\sum_{i,k}\left(\operatorname{sort}(\widetilde z W)_{ik}-q_i\right)^2.$$

投影方向固定，需要 detach。特征的梯度经过中心化、投影和排序；标准差**只在 shape 的分母里 detach**，scale 里的标准差保留梯度。总体标准差、平方尺度误差以及 mean 归约遵循 Algorithm 1。分母使用公式 (2) 的 `+eps`，因此完全坍缩的输入也有有限的前向结果。

`view_invariance(views, num_global=2)` 接收 `[V,N,D]`。前 `G=num_global` 个是全局视图，按样本求它们的均值 `g=views[:G].mean(0)`，返回 `mean((views-g)^2)`，对**所有视图、样本和特征**取平均。`g` 保留梯度，这个函数没有冻结的教师分支。

例如 `[[[0,2]], [[2,0]], [[5,-1]]]`、`G=2` 的全局均值为 `[1,1]`，一致性损失为 `4`。

## 接口约定

输入是有限的 CPU float32/float64 张量，dtype/device 一致；不修改输入，保持输出 dtype。`N>=2`、`D,K>=1`，每个方向非零，`eps` 有限且大于 0，`G` 为 `[1,V]` 内的整数。支持非连续特征张量。校验代码已提供。投影值相同或方差恰好为零时，检查前向结果，不要求数学上唯一的导数。

本题分别实现两个损失组件。toy 使用 `0.5 * invariance + 0.5 * mean_over_views(scale + shape + center)`。编码器、图像增强、投影采样和分布式训练不在实现范围内。

## 小实验

**教学改编。** 三组对齐的 `24×2` 特征从较小方差和非零均值开始，调用你的两个函数优化 80 步，使用固定投影方向。图表显示提交实现的目标和 shape 项，同时独立测量总体标准差与均值平方。这是小规模特征优化实验，不是论文预训练结果的复现。预览用参考实现，T3 使用你的提交。
