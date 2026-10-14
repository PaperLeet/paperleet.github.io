# 参考解与坑点

RMS 是平方均值的平方根，不是方差或 L2 范数。不要去均值，不要使用样本方差修正，也不要让仿射权重参与尺度估计。`mean(..., keepdim=True)` 让结果保持逐 token 广播。

epsilon 的位置是数值契约。本题用平方根内的版本；作者早期代码的外置 epsilon 并不是论文错误，只是不符合本题约定。对零向量，输出为零，输入 Jacobian 却是 `diag(weight)/sqrt(eps)`，不能因此返回一个断开梯度的常量。

令 $s=\sqrt{\sum_{j<k}x_j^2/k+\varepsilon}$，使用从 0 开始的索引，则

$$
\frac{\partial y_i}{\partial x_j}
=w_i\left(\frac{\delta_{ij}}s-\frac{x_i x_j}{k s^3}\mathbf1_{j<k}\right).
$$

尾部输入不参与尺度统计，但尾部输出的损失会通过共享分母影响前缀输入。detach 前缀统计量可能保留正确前向值，却丢掉第二项；解析 VJP 和数值差分都会抓到它。

`k=D` 才必须与全量版本一致。小前缀若代表性不足，可能把尾部放大很多；这不违背公式。教学图刻意选择一个低能量前缀，并不是 pRMSNorm 对任意数据的性能结论。

正 epsilon 下，应验证 `f(a*x, a*a*eps)=f(x, eps)`（$a>0$），而不是把固定 epsilon 的缩放当成精确不变量。RMSNorm 也不具有平移不变性。
