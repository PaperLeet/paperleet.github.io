# DayDreamer: 题解与易错点

用log_softmax得到logq/logp。动力学项冻结整个q，包括概率与logq；表征项只冻结logp。alpha=1只更新prior，alpha=0只更新posterior。不能用一个未detach的KL乘权重代替。

用log_softmax得到logq/logp。动力学项冻结整个q，包括概率与logq；表征项只冻结logp。alpha=1只更新prior，alpha=0只更新posterior。不能用一个未detach的KL乘权重代替。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Dynamics KL wrongly updates the posterior encoder.
- Dynamics KL averages instead of summing independent categorical latents.

## 实验范围

提取DayDreamer继承DreamerV2的KL balancing核心，固定alpha默认0.8；计算free-nats截断之前的KL，不实现RSSM或机器人在线训练。

## 可重复实验记录

1个batch、2个潜变量、每个3类；固定posterior，prior从均匀开始，alpha=0.8，80步梯度下降(lr=1)。要求KL<初始1e-5且最大概率误差<0.002。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
