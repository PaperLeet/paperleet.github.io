# 题解与常见错误

相同协方差的高斯 KL 化简为均值差的加权平方和。时间维求和对应轨迹 KL 上界；batch 维求均值。奖励乘当前 log 密度的 score 项，直接 KL 项保留梯度。

$$
K_{bt}=\frac{\|\mu_{bt}-\mu^{ref}_{bt}\|^2}{2s_t^2},\quad L=\frac1B\sum_b[-\alpha\operatorname{sg}(r_b)\sum_t\ell_{bt}+\beta\sum_t K_{bt}].
$$

Current Gaussian means and KL penalties differentiate. Reference means, scales and terminal rewards are frozen; supplied logp differentiates. State-visitation gradients omitted as in Eq. (9).

KL 不是采样动作 log 比率的单次估计；它是闭式的分布距离。不要对时间求平均，也不要将 KL 一并 detach。

测试依据：标量 KL、时间重复线性缩放、解析奖励与正则梯度、零 KL 端点和在线有正则 bandit。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.035 (70 updates).

```json
{"final_error": 0.00024093245881487226, "initial_error": 0.6944444444444445, "mean": 0.8178113341446054}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
