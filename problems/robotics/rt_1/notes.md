# RT-1: 题解与易错点

将logits转成(B,K,L)，沿最后维归一化后乘原始features。每个学习token是一组空间凸组合，不能沿K归一化，也不应对被汇聚features再次做LayerNorm。

将logits转成(B,K,L)，沿最后维归一化后乘原始features。每个学习token是一组空间凸组合，不能沿K归一化，也不应对被汇聚features再次做LayerNorm。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Softmax normalizes learned tokens rather than spatial positions.
- The learned spatial weights are discarded.

## 实验范围

提取视觉token压缩器；MLP注意力logits作为输入，不训练EfficientNet、Transformer或真实机器人。

## 可重复实验记录

3个空间patch、1个token，目标汇聚值0.8；分数从零开始，100步梯度下降(lr=1)。要求MSE<初始0.1%。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
