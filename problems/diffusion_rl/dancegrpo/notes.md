# 题解与常见错误

把每个 prompt 的一份初噪复制到组内所有样本。每种奖励分别在组内中心化和缩放，再把通道优势相加。终点优势广播到所有转移，用 clipped 目标更新策略。

$$
A_{pg}=\sum_k\frac{R_{pgk}-\mu_{pk}}{\sqrt{\operatorname{mean}_g(R_{pgk}-\mu_{pk})^2}+\varepsilon},\quad L=-\operatorname{mean}_{p,g,t}\min(e^{l_{pgt}}A_{pg},\operatorname{clip}(e^{l_{pgt}},1-c,1+c)A_{pg}).
$$

Noise and reward-derived advantages are frozen. Only current log ratios differentiate in the policy loss; each prompt is normalized independently.

不要跨 prompt 标准化，也不要先把不同单位的奖励相加再标准化。常数奖励通道贡献为零。共享初噪不表示后续 SDE 噪声也共享。

测试依据：手算双通道优势、prompt 隔离、常数通道、共享噪声副本、正负截断和冻结梯度。

## 实验标定

固定种子 42，CPU float64，直接梯度下降；无需数据下载。以下是参考解实际测量，不是预设轨迹。 PyTorch 2.14.1. lr=.03 (70 updates).

```json
{"final_error": 0.00024607032777813887, "initial_error": 1.0, "prompt_groups": 2}
```

验收阈值见 `tests/test_core.py::test_real_toy_experiment`；基线为训练前参数的实际输出。
