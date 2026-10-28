# Octo: 题解与易错点

观察窗口中的padding与动作时域/维度padding含义不同，必须取交集。有效元素占比进入分母，因此追加padding不稀释损失；极端稀疏掩码遵循源码分母下界。

观察窗口中的padding与动作时域/维度padding含义不同，必须取交集。有效元素占比进入分母，因此追加padding不稀释损失；极端稀疏掩码遵循源码分母下界。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Observation-window padding is ignored.
- Padding dilutes the loss because the valid fraction is omitted.

## 实验范围

提取官方动作头的masked_mean及时间/动作padding交集；计算未乘额外action_dim系数的逐坐标平方误差，不实现扩散网络。

## 可重复实验记录

1×2×2×2动作张量，仅2个坐标有效，其余目标设100；40步梯度下降(lr=0.2)。要求loss<初始1e-6，被padding坐标不移动。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
