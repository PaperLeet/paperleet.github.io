# 题解与常见错误

先把 logits 用稳定的 log-sigmoid 转为 log 概率，再用相邻时刻之差构造两条轨迹的偏好比率。差值乘 beta 与总步数 T，最后计算配对 logistic 损失。

$$
d_w=\log\sigma(a^w_{next})-\log\sigma(a^w_{now}),\quad d_l=\log\sigma(a^l_{next})-\log\sigma(a^l_{now}),\quad L=\operatorname{mean}\operatorname{softplus}(-\beta T(d_w-d_l)).
$$

All four classifier evaluations differentiate because both adjacent states contribute to each log probability ratio.

不能直接用原始 logits 的差；log(sigmoid(a)) 的导数是 sigmoid(-a)。不能遗漏当前时刻或总步数因子。

测试依据：标量概率比率 oracle、四分支解析梯度、时间步缩放、极端 logits。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.2 (50 updates).

```json
{"final": 0.05033276942721891, "initial": 0.6931471805599453, "weight": 1.7389346544025606}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
