# 题解与常见错误

按执行顺序遍历调度器，先预测噪声再还原干净样本，再构造下一个噪声级。指定起点之前使用 no_grad；到起点只截断一次状态，后缀内部保留整条链式法则。

$$
\hat x_0^{(i)}=(x_i-\sigma_i\epsilon_\theta(x_i,i))/\alpha_i,\quad x_{i+1}=\alpha_{i+1}\hat x_0^{(i)}+\sigma_{i+1}\epsilon_\theta(x_i,i).
$$

Initial noise and schedules are frozen. Prefix transitions execute without gradients; suffix transitions retain all recurrent state and model paths.

若每步都 detach，前几个后缀步骤无法收到终点奖励。截断只改变梯度，不应改变前向采样值。调度器给的是振幅，不应重复开平方。

测试依据：恒定噪声模型的手算链、不同截断的前向一致性、独立标量递推导数和前缀梯度零。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.05 (45 updates).

```json
{"final": 4.760992065567561e-13, "initial": 1.0, "updates": 45}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
