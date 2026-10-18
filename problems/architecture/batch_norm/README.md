# BatchNorm：批统计量、移动估计与推理

## 1. 论文解决什么问题

Batch Normalization 将可微的批归一化与可训练的缩放、偏置放入网络。训练和推理使用不同来源的统计量。本题依据[算法 1–2 与第 3.2 节](https://arxiv.org/html/1502.03167v3#S3)，实现一个纯函数。论文标题中的 internal covariate shift 是历史动机，本练习不检验 BN 生效原因的因果解释。

## 2. 核心公式与返回值要求

每个通道独立归约 batch 和所有空间位置，`m = N * product(spatial)`：

$$\mu_B=\frac1m\sum_i x_i,\quad v_B=\frac1m\sum_i(x_i-\mu_B)^2,\quad y_i=\gamma\frac{x_i-\mu_B}{\sqrt{v_B+\epsilon}}+\beta.$$

本题明确选择如下移动平均约定：

$$\mu'=(1-a)\mu_{run}+a\mu_B,\quad v'=(1-a)v_{run}+a\frac{m}{m-1}v_B.$$

实现 `batch_norm(...) -> (y, next_mean, next_var)`。输入为至少二维的 `[N,C,*spatial]`，仿射参数和缓冲区为 `[C]`；输出与输入同形状，下一组统计量为 `[C]`。所有 Tensor 的 float32／float64 与 device 一致。不原地修改任何输入。返回的统计量与推理使用的缓冲区均 detach，训练输出所用的批统计量保留梯度。

## 3. 从公式到代码

训练时对 `(0,2,3,...)` 求矩。输出使用分母为 `m` 的总体方差，移动估计使用无偏修正；通道量需要正确 reshape 才能广播。推理时只用给定的 running moments，并返回值不变的统计量，不能重新计算当前 batch 的矩。

## 4. 本题约定

`momentum=a` 是**新 batch 的系数**，不是旧状态的系数；它有限且属于 `[0,1]`。epsilon 必须有限、为正并放在平方根内。训练每通道至少有两个数值；推理允许单样本和空 batch。通道及空间维度为正。形状、样本数量或参数不合法时抛出 `ValueError`。

本题与有状态框架层的接口不同，要求显式返回新缓冲区。移动估计不能把不同优化步骤的计算图连起来；即使调用者给 running buffers 打开了 `requires_grad`，推理也不能对它们求梯度。

## 5. 交互图与实验

固定种子 91，从均值为 3、方差为 4（标准差为 2）的正态分布抽取 80 组 `[32,2]` 样本。学习率 0.08 的 Adam 对照独立标量归一化结果学习 scale=1.5、shift=−0.3；图还展示第零通道的移动估计。每次输出均调用你的实现。

要求最终 MSE 小于 0.003、仿射参数误差小于 0.04。推理时固定样本的输出不能因同批其他样本变化而改变。此实验不复现 ImageNet，不报告加速倍数，也不比较不同归一化方法的模型精度。

## 6. 编写与运行

补全 `batch_norm` 后点击**运行测试**，或使用：

```bash
paperleet test architecture/batch_norm --impl problems/architecture/batch_norm/solution.py
```

测试覆盖标量值、空间归约、无偏移动估计、解析 Jacobian、有限差分、状态梯度隔离、非连续 Tensor、空 batch 推理，以及真实仿射参数训练。

## 7. 延伸

对比 BN 训练时的同批依赖与 GroupNorm 在单样本内求矩的方式。推导固定统计量时可合并成的仿射变换。分析小 batch 对估计方差的影响，再讨论下游模型质量，避免把统计量实验当成精度结论。
