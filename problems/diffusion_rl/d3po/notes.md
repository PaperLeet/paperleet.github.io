# 题解与常见错误

先分别计算胜出和落选轨迹相对参考策略的 log 比率，再在相同时间步相减。每个时间步有独立的 sigmoid 分类任务，最后才对批次与时间求均值。

$$
z_{bt}=\beta[(\ell^w_{bt}-\ell^{w,ref}_{bt})-(\ell^l_{bt}-\ell^{l,ref}_{bt})],\qquad L=\frac1{BT}\sum_{bt}\operatorname{softplus}(-z_{bt}).
$$

Both current preferred and rejected log probabilities differentiate; both reference tensors are detached.

先求时间和再做 softplus 对应另一种目标；正负时间步会抵消。参考模型必须冻结，但落选轨迹的当前模型仍需梯度。

测试依据：不等的逐步 margin 手算、时间重复不变性、参考梯度冻结及极端 margin 稳定性。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.3 (60 updates).

```json
{"final": 0.0839873427692218, "initial": 0.6931471805599453, "minimum_margin": 1.368771445818979}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
