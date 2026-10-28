# 题解与常见错误

对每对分数先求差，再用 softplus 保持大幅度输入下的稳定性。单步预测先冻结已有轨迹状态，再按噪声幅度恢复干净样本。奖励网络参数冻结不等于将奖励值 detach。

$$
L_{rank}=\operatorname{mean}\,\operatorname{softplus}(r_l-r_w),\qquad \hat x_0=(\operatorname{sg}(x_t)-\sigma_t\epsilon_\theta)/\alpha_t.
$$

Ranking differentiates both scores. ReFL stops the rollout latent and differentiates epsilon; the frozen reward model must still propagate gradients to its input.

不要把配对差值先平均再取 logistic；那会让相反偏好互相抵消。alpha 是振幅，不能再开平方。冻结 latent 能切断前缀梯度，epsilon 的梯度应为 -sigma/alpha。

测试依据：独立标量 logistic、极端分数、反解恒等式和解析梯度；T3 先训练奖励评分器再优化生成参数。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. ranking lr=.3 (40 updates); generator lr=.03 (30 updates).

```json
{"final": 0.089454537799872, "initial": 0.6931471805599453, "reward_gain": 1.9853489655118983}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
