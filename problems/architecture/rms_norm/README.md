# RMSNorm 与部分特征归一化

## 1. 问题

实现 `rms_norm` 和 `partial_rms_norm`，理解归一化时保留均值会改变什么，以及只用部分特征估计尺度会带来什么影响。每个 token 独立处理，不需要构建 Transformer。

依据是 [RMSNorm 原论文](https://arxiv.org/abs/1910.07467) 的式 (4) 和第 5 节；作者为 Biao Zhang、Rico Sennrich，发表于 NeurIPS 2019。

## 2. 核心公式

对一个有 $D$ 个特征的向量，本题采用：

$$
s_D(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_D(x)}w_i.
$$

部分特征版本只替换统计量，仍输出全部特征：

$$
s_k(x)=\sqrt{\frac1k\sum_{j=1}^k x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_k(x)}w_i,\quad i=1,\ldots,D.
$$

没有减均值，也没有可学习 bias。`weight` 在归一化之后相乘，不能先乘到统计量中。

## 3. 公式到代码

沿最后一维计算平方均值，保留该维以便广播。pRMSNorm 的分母来自 `x[..., :k]`，分子仍是完整的 `x`。不要 detach 分母：即使只改变尾部输出的损失，前缀特征也会通过分母收到梯度。

可用手算检查：`x=[-3,4,12]` 的全量平方均值是 $169/3$，前两个特征的平方均值是 $25/2$。`k=D` 应恢复全量版本。

## 4. 本题约定

| 项目 | 约定 |
| --- | --- |
| 输入 | `x [..., D]`、`weight [D]`，$D\ge1$，同一浮点 dtype/device |
| 精度 | 有限 float32/float64；保留精度、形状、设备和梯度，不修改输入 |
| epsilon | 正数，放在平方根内：`sqrt(mean_square + eps)` |
| 部分统计 | `k` 是显式整数，$1\le k\le D$；非法值和 bool 抛出 `ValueError` |
| 实现范围 | 用 tensor 运算补全两个函数，不直接调用归一化库 API |

论文的主公式省略了数值稳定项。作者 2019 年的 PyTorch 代码把 epsilon 加在 RMS 外，本题选择平方根内的约定；两种约定在小输入时不同。本题直接传入 `k`，不要求复现论文与代码中百分比取整的差别。

## 5. 交互图与实验

实验提供两个八维向量：一个前缀与整体能量均衡，另一个前缀很小、尾部较大。逐步增大 `k`，观察 pRMSNorm 与全量 RMSNorm 的偏差。偏差不保证对任意输入单调下降；`k=D` 时才必须一致。

另一张图给输入加不同常数，观察输出均值与 RMS。正 epsilon 下，固定 epsilon 的尺度不变性也不是严格恒等式；同时把输入乘 $a>0$、epsilon 乘 $a^2$ 才能精确保持输出。

**来源：** PaperLeet 依据归一化公式设计的无训练教学实验。它不复现机器翻译结果，也不测量论文的硬件加速比。测试和实验都调用你提交的函数，参考预览单独标记。

## 6. 怎么运行

在网页右侧补全代码并运行完整测试。也可在仓库中执行：

```bash
paperleet test architecture/rms_norm --impl problems/architecture/rms_norm/solution.py
paperleet mutate architecture/rms_norm
```

## 7. 延伸

T0 检查接口、精度、非连续输入和空批次；T1 使用手算数值；T2 检查零向量、逐 token 独立性、尺度关系、库对齐、解析梯度与数值差分；T3 检查实际提交函数产生的前缀偏差与平移实验。

可以对照 LayerNorm 模块：常数向量经过 LayerNorm 后的归一化部分为零，而这里仍然非零。再推导 pRMSNorm 的 Jacobian，判断哪些输入坐标会通过尺度统计影响其他输出。
