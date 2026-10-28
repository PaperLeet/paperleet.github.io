# FlashAttention-2 延后归一化与行块 — 题解

## 推导与实现

外层遍历 Q 的行块，使每块独立拥有输出。内层对 K/V 块更新稳定最大值、分母和未归一化分子。最大值变化时同时重缩放旧分母与分子，结束后一次归一化。

$$
m\leftarrow\max(m,\max_j S_j),\quad \ell\leftarrow e^{m_{old}-m}\ell+\sum_j e^{S_j-m},\quad U\leftarrow e^{m_{old}-m}U+e^{S-m}V,\quad O=U/\ell,\quad L=m+\log\ell.
$$

## 边界与梯度归属

q,k:(N,D)，v:(N,V)，返回 O:(N,V) 和 L:(N,)。分数包含 1/sqrt(D)。query_block 与 key_block 为正整数；causal=True 时只允许 key_index<=query_index。每行至少能见自身；最后才对 U 除以 l。所有输出保持梯度。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

对确定性 13 token、4 维 query/key 和 3 维 value 扫描 5 种不整除块大小，分别测试因果与全注意力，比较稠密结果和 LSE。

单头方形注意力的 CPU 规格实现，含因果 mask、尾块、数值稳定性与自动微分；不计 GPU warp 时间。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| maximum_output_error | 8.8817842e-16 |
| maximum_lse_error | 3.5527137e-15 |

## 常见错误

- m1_missing_numerator_rescale: 最大值变化时没有重缩放旧分子
- m2_unscaled_logits: 忘记head维度缩放

[FlashAttention-2](https://arxiv.org/abs/2307.08691) · Section 3.1.1, Algorithm 1; Section 3.3, query-row work partitioning.
