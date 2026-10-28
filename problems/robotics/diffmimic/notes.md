# DiffMimic: 题解与易错点

回放在下一次策略/物理转移前替换偏离过大的状态，因此保留分支可传梯度、回放分支不将梯度传给旧状态。状态匹配loss应由真实轨迹误差计算，不能因为回放而将当前错误记录清零。PD目标角速度为零。

回放在下一次策略/物理转移前替换偏离过大的状态，因此保留分支可传梯度、回放分支不将梯度传给旧状态。状态匹配loss应由真实轨迹误差计算，不能因为回放而将当前错误记录清零。PD目标角速度为零。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- States exactly at the threshold are kept instead of replayed.
- PD control omits velocity damping.
- State matching averages the time horizon instead of summing it.

## 实验范围

实现论文状态误差触发的示范回放、PD力矩和时域状态匹配；实验是标量可微积分器，不是Brax人形机器人复现。

## 可重复实验记录

3个初态、12步标量积分器，dt=0.15，PD增益(2,0.4)，回放阈值2；策略gain从1开始60步梯度下降(lr=0.04)。要求loss<初始90%、最终gain<0，并单独验证阈值0.1回放。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
