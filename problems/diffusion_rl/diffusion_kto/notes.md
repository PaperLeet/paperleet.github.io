# 题解与常见错误

先用无关样本的 log 比率求均值并截断到非负值。喜欢标签保留相对参考的方向，不喜欢标签反转方向。sigmoid 效用以零为中心，外层取负号用于梯度下降。

$$
b=\max(0,\operatorname{mean}\operatorname{sg}(\ell_{unrelated})),\quad z=w\beta(\ell_\theta-\operatorname{sg}(\ell_{ref})-b),\quad L=-\operatorname{mean}(\sigma(z)-1/2).
$$

Only current log probabilities receive gradients. Reference, feedback labels and the unrelated-batch baseline are detached.

基线不是当前训练样本自己的均值；必须来自独立的无关配对。beta 同时缩放 log 比率与基线。此处优化 sigmoid 效用，不是 log-sigmoid 分类损失。

测试依据：逐样本标量 sigmoid、基线截断、正负标签对称性、解析梯度和冻结边界。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.2 (60 updates).

```json
{"final": -0.4760557262293378, "initial": -0.0, "margin": 1.8632436167371014}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
