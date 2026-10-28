# 题解与常见错误

按已完成更新次数推进窗口，边界到达末端后保持。ODE 步只用速度积分；SDE 步加上相应 score 修正与标准差。损失只切出窗口，按窗口长度而不是总轨迹长度平均。

$$
l=\min(\lfloor m/\tau\rfloor s,T-w),\quad W=\{l,\ldots,l+w-1\}.\quad L=-\operatorname{mean}_{b,t\in W}\min(e^{\ell_{bt}}A_b,\operatorname{clip}(e^{\ell_{bt}},1-c,1+c)A_b).
$$

Rollout states are detached, velocity remains differentiable. Advantages are frozen; loss gradients are zero outside the selected window.

窗口不会循环回开头。窗口外步骤仍用于生成终点奖励，但不参与策略目标；用总步数作分母会稀释梯度。时间是从噪声向数据递减，dt 为负。

测试依据：窗口边界手算、ODE/SDE均值与方差、未选步骤梯度零、子窗口归约和在线混合采样。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.15 (60 updates).

```json
{"early_error": 0.8417650037927409, "last_start": 2, "late_error": 0.0029253598150183547}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
