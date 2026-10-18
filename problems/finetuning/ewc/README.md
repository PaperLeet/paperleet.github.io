# 95. EWC：用参数重要性约束遗忘

## 问题

模型先学任务 A，再学任务 B。新任务的梯度可能移动旧任务依赖的参数。EWC 根据各参数对旧任务的重要性，惩罚它们偏离旧解的程度。

实现 `ewc_penalty(theta, anchor, fisher_diag, strength)`。`anchor` 是学完 A 后保存的参数；`fisher_diag` 是**已经给定**的非负 Fisher 对角权重。本题不要求估计 Fisher，也不要求加载大模型。

## 核心公式

[Kirkpatrick 等，§2 式 (3)](https://arxiv.org/html/1612.00796v2#S2.E3) 将新任务损失与旧任务惩罚相加：

\[
L(\theta)=L_B(\theta)+P(\theta),\qquad
P(\theta)=\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta^*_{A,i})^2.
\]

你只返回 \(P\)，调用者另加 \(L_B\)。\(\lambda\) 对应 `strength`；\(F_i\) 越大，同样位移的代价越高。它是对**所有参数求和**，不是按参数数目求平均。不要再乘旧任务的样本数。

## 公式到代码

```python
ewc_penalty(theta: Tensor, anchor: Tensor,
            fisher_diag: Tensor, strength: float) -> Tensor
```

例如 `theta=[2,-1]`、`anchor=[1,1]`、`fisher_diag=[2,0.5]`、`strength=3`，返回标量 `6`。对当前参数的梯度为 `[6,-3]`；旧参数和 Fisher 快照不接收梯度。

先识别哪些量是正在训练的参数，哪些量是冻结的记录，再计算逐元素位移与标量归约。`detach` 快照不等于冻结 `theta`。

## 本题约定

- 三个输入 Tensor 形状、dtype、device 相同，位于 CPU，dtype 为 float32/float64；值有限，Fisher 非负。输入保证合法，不要求写参数校验。
- 支持任意同形状参数块，包括标量、空 Tensor 和非连续切片；不做广播。返回 `shape == ()` 的 Tensor，保持 `theta` 的 dtype/device，不修改任何输入。
- 只有 `theta` 保留梯度。即使 `anchor` 或 `fisher_diag` 的 `requires_grad=True`，也把它们视为固定快照。
- `strength` 是有限非负 Python 数值。零强度、零 Fisher 坐标、空输入贡献可反向传播的零；零 Fisher 坐标上的极大有限位移也不能先平方成无穷，再用 `0 * inf` 得到 NaN。非零贡献的运算在 dtype 可表示范围内。
- 不包含新任务损失、Fisher 估计、多个历史任务的合并或 replay buffer。

## 交互图与实验

**My run results** 用你实现的惩罚函数参与每一步新任务更新。实验先用 SGD 训练只有两个参数的线性模型，再复制同一旧解，比较 `strength=0` 与 `strength=1` 学习任务 B 的轨迹。

A 的四个输入为 `[-2,0],[-1,0],[1,0],[2,0]`，标签为 `2*x0`。B 的四个输入为 `[-1,0],[1,0],[0,-1],[0,1]`，标签为 `-2*x0+2*x1`。两任务的预测均为 `x @ theta`，训练数据项为 `MSE/2`。单位方差高斯观测模型给定 A 的 Fisher 对角 `E[x_i²]=[2.5,0]`，无需采样估计。A 训练 60 步，学习率 0.2；两个 B 分支各训练 160 步，学习率 0.1；CPU float64，数据完全确定。

图中同时显示旧、新任务的 MSE：本例的 EWC 分支更保留 A，也牺牲了一部分 B 的拟合。第二个参数的 Fisher 为零，仍能学习 B。图表测量这两个合成任务，不是 held-out 泛化、论文 MNIST/Atari 复现或大模型效果，也不保证任意任务都不会遗忘。参考图与做题者实际运行结果分开显示。

## 运行方式

在网页编辑器完成函数，点击 **Run tests**，再查看 Lab；官方参考实现可在本地验证：

```bash
paperleet test finetuning/ewc --impl problems/finetuning/ewc/solution.py
paperleet mutate finetuning/ewc
```

T0 检查标量和 dtype；T1 用独立 Python 求和核对公式；T2 检查解析梯度、冻结、零边界、加法性和输入保护；T3 实际训练并比较独立求出的二次目标最优点。

## 延伸

为什么统一的 L2 系数不能表达参数重要性的差别？若增大 `strength`，旧任务与新任务的误差如何权衡？Fisher 的估计方法、任务边界以及多个历史任务如何累计，是后续持续学习问题，不属于本题接口。
