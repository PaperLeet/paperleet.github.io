# 两种方差估计与两种执行模式

训练对当前 batch 归一化，方差分母是 `m`。用独立抽样的小批次估计总体方差时，再乘 `m/(m-1)`。把无偏修正用于训练输出，会改变算法 1；从移动方差中漏掉修正，则改变本题选定的推理估计方式。

对一个通道，令 `z=(x-mean)/sqrt(v+eps)`、`q=upstream*gamma`，有：

$$\frac{\partial L}{\partial x_i}=\frac{q_i-\operatorname{mean}(q)-z_i\operatorname{mean}(qz)}{\sqrt{v+\epsilon}}.$$

detach 均值与方差会丢掉两个修正项。scale、bias 的梯度分别在 batch／空间位置求和 `upstream*z` 与 `upstream`。相反，移动估计应有意 detach，避免后续 batch 反传到以前的计算图。

六个 mutant 混淆训练方差分母、漏掉移动无偏修正、反转 EMA 系数、detach 实时统计量、把 epsilon 放到根号外，或在移动状态里保留计算图。固定统计量下的同批独立性是结构性质，不是模型精度结论。有限随机 batch 的移动方差不必精确等于 4，因此只展示估计曲线，不据此要求不成立的精确总体值。
