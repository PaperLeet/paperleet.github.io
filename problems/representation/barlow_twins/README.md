# Barlow Twins：交叉相关与冗余消除

## 1. 问题

同一样本的两种增强应保留相同语义，但把所有样本编码成相同向量也会满足简单的一致性目标。[Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html) 用特征间的交叉相关矩阵同时约束两件事：对应特征彼此一致，不同特征减少冗余。实现 `cross_correlation(z1, z2, eps=1e-5)` 和 `barlow_twins_loss(z1, z2, lambd=.0051, eps=1e-5)`。

输入是已经完成投影的成对向量，两组都是 `[B,D]`，同一行来自同一样本。输出分别为 `[D,D]` 矩阵和零维损失。这里没有负样本队列、预测器或 stop-gradient。

## 2. 核心公式

每个视图独立沿 batch 轴计算每个特征的均值与 **population variance**。为对应[官方代码](https://github.com/facebookresearch/barlowtwins/blob/main/main.py)中的 `BatchNorm1d(affine=False)` 训练行为，采用：

$$
\mu_i^{(v)}=\frac1B\sum_b z_{bi}^{(v)},\quad
\sigma_i^{2(v)}=\frac1B\sum_b(z_{bi}^{(v)}-\mu_i^{(v)})^2,\quad
\hat z_{bi}^{(v)}=\frac{z_{bi}^{(v)}-\mu_i^{(v)}}{\sqrt{\sigma_i^{2(v)}+\epsilon}}.
$$

然后构造完整交叉相关矩阵，行索引属于第一视图，列索引属于第二视图：

$$
C_{ij}=\frac1B\sum_b\hat z_{bi}^{(1)}\hat z_{bj}^{(2)},\qquad
\mathcal L=\sum_i(C_{ii}-1)^2+\lambda\sum_i\sum_{j\ne i}C_{ij}^2.
$$

这对应论文式 1–2 的目标，以及官方代码显式的 BN 数值稳定约定。固定正数 `eps` 会让相同非恒定特征的相关系数略小于 1。两项均为**求和**；非对角项遍历上下两个三角，矩阵一般不对称。

手算：若两组输入相同，每列均值为 0、population variance 为 1，且各列正交，则 `C=I/(1+eps)`。两个完全重复的这样的特征会产生两项相同的非对角惩罚，而不是一项。

## 3. 公式到代码

1. 沿 `dim=0` 求均值并中心化，保留 `[B,D]` 布局。
2. 对中心化值的平方沿 `dim=0` 求均值，先加 `eps`，再开方并相除。
3. 用 `normalized1.T @ normalized2 / B` 得到 `[D,D]`。这不是样本之间的 `[B,B]` 相似度矩阵。
4. 对角线以 1 为目标，其余元素以 0 为目标。按公式分别求平方和，再乘 `lambd` 合并。

`barlow_twins_loss` 应调用当前文件里的 `cross_correlation`。两组输入都必须保留梯度，包括标准化时的均值和方差。提供的 `_check_inputs`、`_check_scalar` 负责输入检查，可直接保留。

## 4. 本题约定

- `B>=2, D>=1`；两组输入形状、device、float32/float64 dtype 相同，且所有元素有限。保留 dtype 和 device，支持非连续视图，不修改输入。
- `eps` 是有限正 Python int/float；`lambd` 是有限非负 Python int/float；不接收 bool 或 Tensor 参数。
- 方差和最终矩阵均除以 `B`。无 running statistics、affine 参数或跨进程聚合；这是单 batch 的教学核心。
- `D=1` 时没有非对角项；`lambd=0` 时只计算对角目标。恒定特征标准化后为零；双视图全部恒定时损失为 `D`，在精确塌缩点梯度也为零。
- 固定 `eps` 下，任意缩放不再严格保持相关系数。把两组输入同时乘非零常数 `s`，并把 `eps` 乘 `s²`，才得到严格相同的矩阵；`s=0` 会使 `eps=0`，违反输入契约。

测试包含独立标量 covariance oracle、完整矩阵 VJP 与损失对两组输入的有限差分、独立平移、成对行/列置换、视图互换、重复 batch、单特征与近零方差。

## 5. 交互图与 CPU 实验

实验室展示实际训练目标、独立计算的非对角相关平方和，以及训练前后九个矩阵元素。训练直接学习两张 `16×3` 向量表，共 96 个参数。确定性的三角函数基底与不同混合矩阵产生初始相关特征，使用 Adam、学习率 `.04`、`lambd=.2`、`eps=1e-5`，执行 **120 次真实更新**，每 10 步记录一次。

T3 调用你的损失进行训练，用独立 Python 标量计算核对最终矩阵与目标。验收要求最终损失小于初始值的 1%、对角误差小于 `.005`、非对角平方和小于 `.01`。`.2` 是为小实验选择的教学权重；函数默认 `.0051` 对应官方设置。这是合成向量表实验，不训练图像编码器，不等同于论文 ImageNet 结果。

## 6. 运行方式

网页中补全两个函数后选择 **Run tests**。核心运算与全部测试支持 CPU/WASM，无需 GPU、数据下载或预训练模型。CLI：

```bash
paperleet test representation/barlow_twins --impl problems/representation/barlow_twins/solution.py
paperleet mutate representation/barlow_twins
paperleet ci --problem representation/barlow_twins
paperleet vizdata representation/barlow_twins
```

## 7. 延伸

与 [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) 对照：SimCLR 区分样本级正负配对，本题约束特征级矩阵。与 [VICReg](https://arxiv.org/abs/2105.04906) 对照：VICReg 分开约束样本一致性、特征方差和单视图 covariance，不能把其 `B-1` 样本方差约定直接替换到本题。

继续阅读[原论文式 1–2、Algorithm 1](https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf)与[官方 `BarlowTwins.forward`](https://github.com/facebookresearch/barlowtwins/blob/main/main.py)。思考：当 `D>B-1` 时，中心化表示的秩如何限制 `C` 达到单位矩阵？为什么有塌缩惩罚并不保证能从精确常数初始化处恢复？
