# TD-MPC2: 题解与易错点

SimNorm在每个小组内归一化，不能让整个latent共享一个softmax。轨迹权重沿精英E维，对整条时域共享。官方temperature乘价值；方差围绕加权均值，最后再夹标准差。公式用精确归一化，省略源码分母的1e-9数值保护。

SimNorm在每个小组内归一化，不能让整个latent共享一个softmax。轨迹权重沿精英E维，对整条时域共享。官方temperature乘价值；方差围绕加权均值，最后再夹标准差。公式用精确归一化，省略源码分母的1e-9数值保护。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Temperature divides elite values instead of multiplying them.
- Second moment is mistaken for centered variance.

## 实验范围

提取SimNorm与MPPI已选精英的加权矩更新；候选轨迹与价值作为输入，不学习动力学/Q网络或执行真实MPC。

## 可重复实验记录

2步1维动作、25点二维网格，选择8精英，12轮更新；temperature=5，std范围[0.03,1]。初始均值0，要求终误差<初始3%；另检查两个3维单纯形质量和。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
