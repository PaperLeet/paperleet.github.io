# BYOL：归一化回归与动量目标

## 1. 问题

实现 `byol_loss(p1, p2, z1, z2, eps=1e-12)` 和 `ema_update(online, target, momentum=.99)`。`p1,p2` 是在线分支两视图的预测，`z1,z2` 是目标分支投影。学习在线预测去匹配另一视图的目标，并用在线参数的指数移动平均更新目标参数。

[BYOL 原论文](https://arxiv.org/abs/2006.07733)第 3.1 节与 Algorithm 1 是依据。与 SimSiam 题相比，本题使用归一化平方距离，并单独实现参数 EMA；目标网络不参与梯度更新。

## 2. 核心公式

本题沿用官方 JAX helper 的**平方范数**下界：

$$
\bar v_i=\frac{v_i}{\sqrt{\max(\sum_d v_{id}^2,\epsilon)}}.
$$

损失按官方实现对两个方向求和，再对 batch 求均值：

$$
L=\frac1B\sum_i\left[\|\bar p_{1i}-\overline{\mathrm{sg}(z_{2i})}\|_2^2
+\|\bar p_{2i}-\overline{\mathrm{sg}(z_{1i})}\|_2^2\right].
$$

不触发 epsilon 下界时，单方向距离为 `2−2*cosine`；两方向均完全反向时总损失为 `8`，不是 `4`。使用平方距离表达式也定义了零向量处的行为。

$$
\theta_{\mathrm{target}}^{new}=m\theta_{\mathrm{target}}^{old}+(1-m)\theta_{\mathrm{online}}.
$$

`m=1` 保留旧目标，`m=0` 复制在线参数。

## 3. 公式到代码

四组向量均沿特征轴归一化；停止 `z1,z2` 的梯度，交叉匹配，先对特征求和，再对样本求均值。不要多除以 2 或特征维数。`ema_update` 接收参数张量，不归一化参数；两输入都从计算图断开，返回新存储。训练循环先优化在线分支，再调用 EMA 更新目标。

## 4. 本题约定

损失输入是同形有限 `[B,D]`，`B,D>=1`；EMA 接受任意相同非空形状。所有输入共享 float32/float64 dtype 与 device，支持非连续输入，不允许原地修改。损失返回零维张量，EMA 返回独立存储且不需要梯度，端点也不能直接返回输入别名。

`eps` 为有限正 Python 数；它作用于平方范数，默认 `1e-12` 相当于范数下界 `1e-6`。`momentum` 为 `[0,1]` 内有限 Python 数，bool 和 Tensor 标量均不接受。校验辅助函数已提供。

测试覆盖独立标量目标、两预测侧有限差分、两目标侧停止梯度、视图交换、成对置换、正缩放、近零向量，以及 EMA 的方向、端点与存储独立性。

## 5. 交互图与 CPU 实验

提供两张 `6×3` 在线表和共享 `3×3` 线性预测头，共 45 个在线参数。初始化为固定三角函数向量，目标表复制在线表。SGD 学习率 `.18`，执行 80 次真实更新；每次在线更新后，使用 `m=.9` 更新对应目标表，每 8 步记录回归损失与目标滞后。独立标量公式逐步核对损失和 EMA。

验收要求损失小于 `.05` 且低于初始值的 5%，目标累计移动大于 `.1`，损失误差小于 `1e-11`、EMA 误差小于 `1e-12`。CPU float64、框架种子 42。该表格实验呈现预测与移动目标的协作，不是共享图像编码器预训练，也不提供普遍的防塌缩保证。

## 6. 运行方式

```bash
paperleet test representation/byol --impl problems/representation/byol/solution.py
paperleet mutate representation/byol
```

练习时把 `--impl` 换成补全的 starter。网页运行全部 CPU/WASM 测试，无需图像数据、GPU 或预训练权重。

## 7. 延伸与来源

对比 SimSiam：`detach` 本身不改变 forward 数值，但决定梯度归属；BYOL 再通过 EMA 产生随训练缓慢变化的目标。考虑不同 momentum 下目标滞后如何改变，区分在线预测参数与目标编码参数，不能对预测头输出直接做参数 EMA。

来源：[论文](https://arxiv.org/html/2006.07733v3)、[官方归一化及回归 helper](https://github.com/google-deepmind/deepmind-research/blob/master/byol/utils/helpers.py)、[官方双向损失及训练流程](https://github.com/google-deepmind/deepmind-research/blob/master/byol/byol_experiment.py)。本题代码、测试与表格实验独立编写。
