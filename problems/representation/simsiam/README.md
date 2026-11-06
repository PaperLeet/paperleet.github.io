# SimSiam：停止梯度与双向预测

## 1. 问题

SimSiam 用一个视图的预测向量匹配另一个视图的投影向量。完成 `negative_cosine_similarity(prediction, target)` 与 `simsiam_loss(p1, p2, z1, z2)`，掌握这种无需负样本、无需动量编码器的目标，以及决定其优化方向的停止梯度操作。输入已经是网络输出，不需要编写编码器或预测头。

依据：[原论文](https://arxiv.org/abs/2011.10566)第 3 节式 (1)–(4) 与 Algorithm 1。

## 2. 核心公式

对 `[B,D]` 张量逐行归一化，定义本题的零向量约定：

$$
\hat v_i=\frac{v_i}{\max(\lVert v_i\rVert_2,10^{-12})}.
$$

单方向损失和对称损失分别为：

$$
D(p,z)=-\frac1B\sum_{i=1}^B \hat p_i^\top\widehat{\operatorname{stopgrad}(z_i)},
\qquad L=\frac{D(p_1,z_2)+D(p_2,z_1)}2.
$$

当两侧向量的范数都至少为 `1e-12`、不触发范数下界时，同向给出 `-1`，反向给出 `+1`；正交向量给出 `0`。负损失是正确结果。先对特征求和，再对样本求平均；两方向的均值还需除以 2。

## 3. 公式到代码

1. 在单方向函数内断开 `target` 的梯度，保留 `prediction` 的计算图。
2. 沿 `dim=1` 对两侧分别做 L2 归一化。
3. 计算逐行负点积并平均。
4. 对称函数交叉匹配 `p1↔z2`、`p2↔z1`。

`detach()` 不会让共享编码器整体停止学习。当 `p1=h(z1)` 时，`z1` 仍可通过 `p1` 的预测路径收到梯度；仅当它作为另一个方向的目标时不回传。不要对预测分支、最终损失或共享编码器统一断图。

## 4. 本题约定

所有输入形状相同，均为有限的 `[B,D]`，`B>=1, D>=1`，共享 float32/float64 dtype 与 device。返回零维标量，保持 dtype/device，不改写输入，支持非连续张量。已提供的校验函数拒绝空轴、广播形状、混合 dtype 和 NaN/Inf。

归一化明确采用 `max(norm,1e-12)`，包括零向量和接近零的向量。零预测与非零目标的损失为 0，但预测梯度通常不为零；不要用“零损失”推断“零梯度”。这一 epsilon 是本练习的数值约定，论文主要讨论非零表征。

测试使用独立 Python 标量计算、预测端有限差分、目标无梯度检查、共享路径检查、视图交换、成对置换、逐行正缩放和零向量边界。不会要求网络结构、负样本队列或 EMA。

## 5. 交互图与 CPU 实验

提供两张独立的 `4×3` 预测表和两张固定目标表，使用 SGD、学习率 `2.0`，实际更新 80 步，每 4 步记录一次。24 个预测参数学习交叉目标；目标也被交给优化器，因此遗漏停止梯度会造成目标移动。图中损失来自待测函数，余弦诊断来自独立标量计算。

验收要求最终平均余弦大于 `.98`、较初始提升超过 `.5`、目标完全不移动、损失与独立标量误差小于 `1e-11`。初始化为固定非对称向量，框架种子 42，CPU float64。该实验用于观察梯度路径，不等同于共享图像编码器预训练；不能据此证明任意模型都不会塌缩。

## 6. 怎么运行

```bash
paperleet test representation/simsiam --impl problems/representation/simsiam/solution.py
paperleet mutate representation/simsiam
paperleet vizdata representation/simsiam
```

在网页编辑器补全两个函数后运行测试；CPU/WASM 使用相同的完整测试与实验，无需 GPU 或数据下载。

## 7. 延伸

对比 InfoNCE/SimCLR 的负样本分类、MoCo 的动量目标与队列，以及 VICReg 的显式方差和协方差约束。SimSiam 的目标值在移除停止梯度后完全不变，但导数改变，说明只核对 forward 数值不足以验证实现。

来源：[论文 Algorithm 1](https://arxiv.org/html/2011.10566v1)、[官方模型的目标分支](https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py)、[官方双向训练损失](https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py)。题面、测试、标量 oracle 与教学实验由 PaperLeet 独立编写。
