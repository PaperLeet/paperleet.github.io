# ACT: 题解与易错点

官方实现按旧到新索引使用负指数；因此正decay给较早预测更大权重。显式valid替代源码的全零哨兵，避免把合法零动作丢掉。KL中的logvar是对数方差。

padding不应占据指数序号；所有动作坐标共享一组时间权重。KL遗漏维度求和会让潜变量数改变正则强度。

## 测试捕获的错误

- Temporal weights favor newest predictions instead of the declared official ordering.
- KL incorrectly averages latent dimensions.

## 实验范围

实现ACT的两个张量核心；输入已经对齐到同一执行时刻，省略视觉Transformer、动作块网络和遥操作系统。

## 可重复实验记录

64个正弦目标、5个重叠预测，decay=0.01；基线为最新预测。另将4×3风格均值用40步梯度下降(lr=0.5)推向标准正态。要求集成MSE<最新预测10%，最终KL<初始1e-4。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
