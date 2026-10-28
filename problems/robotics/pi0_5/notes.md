# π₀.₅: 题解与易错点

prefix仅看prefix；FAST看prefix与当前之前FAST；expert看prefix与所有expert。FAST包含干净动作标签，若允许expert读取，会在流匹配训练中泄漏答案。query与key两侧都需屏蔽padding。

prefix仅看prefix；FAST看prefix与当前之前FAST；expert看prefix与所有expert。FAST包含干净动作标签，若允许expert读取，会在流匹配训练中泄漏答案。query与key两侧都需屏蔽padding。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Action-expert queries can read clean FAST action labels.
- The continuous action expert incorrectly uses causal rather than bidirectional attention.

## 实验范围

实现论文附录E/Figure18的训练注意力连接；prefix、离散FAST标签和连续动作expert的内容由调用者给定，省略模型预训练。

## 可重复实验记录

8个token按3 prefix、2 FAST、3 expert分组；均匀合法注意力，FAST扰动幅度0/1/10/100/1000。expert响应必须为0，prefix扰动响应>0.5；这是前向信息流实验，无训练器。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
