# ResNet：残差分支与投影捷径

## 1. 论文解决什么问题

更深的网络即使能表示较浅网络的解，也可能更难优化。ResNet 让一个块学习相对捷径的修正，而非从零学习整个映射。本题依据[第 3.2 节、式 (1)–(2) 与图 2](https://arxiv.org/html/1512.03385v1#S3.SS2)，实现矩阵形式的两层残差块。

本题选取原论文的**相加后激活**结构。卷积、BatchNorm、bottleneck 和完整 ResNet 网络不在范围内；向量回归实验明确属于教学改编。

## 2. 核心公式与返回值要求

使用右乘权重：

$$h=\operatorname{ReLU}(XW_1),\quad r=hW_2,\quad s=X\text{ 或 }XW_s,\quad Y=\operatorname{ReLU}(r+s).$$

实现 `residual_block(x, w1, w2, projection=None) -> Tensor`：

- `x` 为 `[B,D_in]`，`w1` 为 `[D_in,H]`，`w2` 为 `[H,D_out]`。
- 返回 `[B,D_out]`，保留输入一致的 float32／float64 精度与 device。
- `projection=None` 使用恒等捷径，此时必须 `D_in == D_out`；否则使用给定的 `[D_in,D_out]` 投影，即使它是方阵。
- 保留输入、两层权重与投影的梯度，不修改输入。
- 特征维度为正，允许空 batch。形状不兼容抛出 `ValueError`，不能靠广播修补捷径。

## 3. 从公式到代码

先求第一层激活，再求第二层线性输出；单独构造捷径，相加之后再做 ReLU。第二层残差可以为负，先把它截断会改变模型。本题没有需要补充的偏置或归一化层。

## 4. 本题约定

原论文在相加**之后**激活，因此零残差返回的是 `ReLU(x)`，对任意有符号输入不等于恒等函数；正输入才精确恢复恒等。后来的 pre-activation ResNet 属于另一种设计。

本题权重按 `x @ w` 存放，与 `nn.Linear.weight` 的方向不同。投影是可训练的捷径，不是可以 detach 的形状调整；输入梯度需要累加两条路径。数值梯度测试避开 ReLU 的不可微位置。

## 5. 交互图与实验

四个块在 32 个正值三维样本上学习 `target = x + 0.2*x²`，隐藏宽度为 8，使用学习率 0.02 的 Adam 实际更新 80 步。第二层从零初始化，起点是正输入上的恒等映射。曲线来自每个块调用你实现的函数；参考曲线会单独标注。

这是合成回归，不代表 CIFAR／ImageNet 复现，也不说明所有任务上都有加速或精度优势。实验固定种子 73、float64，要求最终 MSE 小于 0.003、损失下降至少 95%，且学到非零残差。

## 6. 编写与运行

在编辑器中补全 `residual_block`，点击**运行测试**。本机命令：

```bash
paperleet test architecture/residual_block --impl problems/architecture/residual_block/solution.py
```

T0 检查接口，T1 用独立标量点积核对，T2 检查激活顺序、投影、梯度和布局，T3 执行真实训练。

## 7. 延伸

在同一宽度下比较恒等与投影捷径。推导两处激活都处于正区间时的 Jacobian，再分析分段变化。完整复现前还需阅读论文的卷积结构与归一化设置，它们不属于这个块的实现范围。
