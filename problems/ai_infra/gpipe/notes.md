# GPipe 填充排空与微批次梯度 — 题解

## 推导与实现

前向依赖来自左侧分段与同段前一微批次；反向从最后分段和最后微批次排空。累计每块平方误差之和，最后按样本数归约，不能平均各块均值。

$$
F_{s,m}=s+m,\quad B_{s,m}=K+M-1+(K-1-s)+(M-1-m),\quad L=\frac{1}{N}\sum_c\sum_{i\in c}(x_i^T w-y_i)^2.
$$

## 边界与梯度归属

stages、microbatches 和 chunk_size 为正整数。事件为 (phase, stage, microbatch, start)，phase 为 F/B；单事件占一个时隙。先完成所有前向，再按逆微批次顺序反向。x 为 (N,D)、y 为 (N,)、w 为 (D,)，允许最后一块较短；总损失除以 N。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

用 11 个确定性样本、4 维线性模型、学习率 0.08 的 35 步 SGD，对比整批和大小 4 的微批次轨迹；同时计算 4 分段在不同微批次数下的有效时隙率。

等时分段、同步更新的 CPU 调度模型与线性回归；不模拟通信、BatchNorm或重计算。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.97372915 |
| final_loss | 0.0019951675 |
| trajectory_error | 5.5511151e-17 |
| utilization_first | 0.25 |
| utilization_last | 0.84210526 |

## 常见错误

- m1_chunk_mean: 把样本均值替换为微批次数归一化
- m2_overlap_backward: 反向阶段提前启动，破坏填充排空屏障

[GPipe](https://arxiv.org/abs/1811.06965) · Section 2.2, Algorithm; Figure 2, synchronous microbatch pipeline.
