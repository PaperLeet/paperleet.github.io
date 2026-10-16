# 怎样理解 GroupNorm

归一化范围就是算法本身：同一个样本、连续的一组通道、这些通道的全部空间位置。reshape 只应把这个集合显式表示出来，不能混入其他样本，也不能把通道改成交错分组。

记上游导数为 $u_i$，$s=\sqrt{v+\varepsilon}$，$z_i=(x_i-\mu)/s$，$q_i=u_i\gamma_{c(i)}$。同一组的输入梯度为：

$$
\frac{\partial L}{\partial x_i}
=\frac{q_i-\operatorname{mean}(q)-z_i\operatorname{mean}(qz)}s.
$$

式中的均值仍只在同一样本与同一组内计算。逐通道缩放梯度为该通道跨样本、跨空间的 $u_i z_i$ 之和；偏置梯度为对应 $u_i$ 之和。因此 detach 统计量可能完全不影响前向数值，却会破坏 Jacobian。

单元素组中心化后为零，输入梯度的直接路径与均值路径互相抵消。epsilon 保证分母有限，不会把总体方差的分母变成 $m-1$。

`G=1` 与 `G=C` 改变的是统计范围，仿射参数始终逐通道。前者与全层 LayerNorm 共享统计范围，但一般 LayerNorm 层的仿射参数形状可能不同。

批次组成实验验证的是结构性独立性。提供的 batch 统计基线直接使用当前激活，不包含滑动平均；它展示训练态批次依赖，不代表训练后模型准确率的排名。参考[论文公式](https://arxiv.org/html/1803.08494v3#S3)及[官方算子约定](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)。
