# 题解与常见错误

保留原始生成样本的奖励梯度。构造每个额外噪声样本时冻结原始图，再用可训练去噪器恢复样本并计算奖励。所有奖励先按样本求均值，再按原始与额外分支平均。

$$
z_k=\alpha\operatorname{sg}(x_0)+\sigma\operatorname{sg}(\epsilon_k),\quad \hat x_k=(z_k-\sigma\epsilon_\theta(z_k))/\alpha,\quad L=-\frac{r(x_0)+\sum_{k=1}^K r(\hat x_k)}{K+1}\text{ averaged over batch}.
$$

Original generated samples retain their reward gradient. Re-noising uses generated.detach() and frozen noises; additional gradients reach only the denoiser. Reward parameters are supplied/frozen by caller but input gradients remain live.

重新加噪时若不冻结生成图，会错误地多次回传原始采样链。只冻结重加噪的输入，不能冻结整个重建结果。平均约定对应论文正文的平均梯度描述，和伪码求和相差固定学习率因子。

测试依据：常数去噪器的手算目标、原图梯度的 1/(K+1) 系数、去噪器解析梯度、噪声顺序不变性。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.08 (55 updates).

```json
{"final": 0.07564035181237105, "initial": 1.015, "weight": 0.6896500813549452}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
