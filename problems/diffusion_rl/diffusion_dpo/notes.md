# 题解

将累计 alpha 变为 `(B, 1, ..., 1)`，信号乘它的平方根，噪声乘其补数的平方根。共享噪声带来一个直接可检验的不变量：

$$
x_t^w-x_t^l=\sqrt{\bar\alpha}(x^w-x^l).
$$

将所有非 batch 维展平，再对特征取误差均值。当前模型与参考模型分别计算“偏好误差减去非偏好误差”。当前差值比参考差值小时，偏好间隔应该为正，因此 `beta/2` 前面带负号。用 `softplus(-margin)` 可稳定处理置信度很高但排序错误的样本。

令第 $i$ 对具有 $D$ 个特征，$s_i=\sigma(-m_i)$，梯度为

$$
\frac{\partial L}{\partial\epsilon^w_{\theta,i,d}}
=\frac{\beta s_i}{BD}(\epsilon^w_{\theta,i,d}-\epsilon_{i,d}),
\qquad
\frac{\partial L}{\partial\epsilon^l_{\theta,i,d}}
=-\frac{\beta s_i}{BD}(\epsilon^l_{\theta,i,d}-\epsilon_{i,d}).
$$

参考预测和目标噪声定义固定比较基准，应显式 detach。特征均值保证复制坐标不会改变损失，偏好对均值保证复制 batch 不会改变损失。

仿射去噪器实验通过增大偏好间隔降低损失，但不保证偏好样本的绝对重建误差每一步都下降。这正是优化误差差值的含义。
