# 题解与常见错误

先用同方差高斯闭式 KL 测量策略分布变化。只在优势与 log 比率方向一致且散度严格超过阈值时屏蔽。屏蔽后的比率在 exponentiation 之前替换，防止零乘无穷大。reference KL 始终参与。

$$
D=\|\mu_\theta-\mu_{old}\|^2/(2s^2),\quad M=1-\mathbf1[D>\delta\ \land\ ((A>0\land l>0)\lor(A<0\land l<0))],\quad L=\operatorname{mean}[-M e^l A+\beta D_{ref}].
$$

Only log_ratios and reference_kl differentiate in loss. Advantages and divergence decisions are frozen. The mask is zero only for favorable-direction updates beyond the divergence threshold; blocked ratios are replaced before exponentiation.

散度大并不意味着所有更新都该屏蔽；纠正方向必须保留。边界 D=delta 或 ratio=1 不屏蔽。掩码的梯度为零，而参考 KL 的梯度不能被掩码关掉。

测试依据：四种优势/比率方向的真值表、严格阈值边界、屏蔽极端比率、解析梯度以及在线高斯训练。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.06 (65 updates).

```json
{"final_error": 3.293515663401782e-06, "initial_error": 0.8233882030178326, "updates": 65}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
