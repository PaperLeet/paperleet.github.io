# 题解与常见错误

每个转移的坐标共同构成一个动作，因此先对特征维求 log 密度之和。旧策略和采样动作保持固定。逐步概率比乘终点优势，对正负优势分别由 min 选择保守分支。

$$
\ell=-\frac12\sum_d[(a_d-\mu_d)^2/s^2+2\log s+\log(2\pi)],\quad L=-\operatorname{mean}_{b,t}\min(e^{\ell-\ell_{old}}A_b,\operatorname{clip}(e^{\ell-\ell_{old}},1-e,1+e)A_b).
$$

Gaussian actions and std are detached; current means differentiate. PPO old log probabilities and advantages are detached. Very large unfavorable log ratios may overflow; finite tested ratios stay within the declared numerical domain |current-old|<=40.

不能将特征维 log 密度取平均，也不能把整条轨迹比率相乘后再截断。负优势下 clip 下界限制过度降低概率的收益。

测试依据：独立高斯标量密度、正负截断手算、score梯度、冻结采样、复制时间步的归约不变性。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.08 (60 updates).

```json
{"final_error": 1.7610391745268231e-06, "initial_error": 1.0, "updates": 60}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
