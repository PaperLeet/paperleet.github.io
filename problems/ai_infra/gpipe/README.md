# GPipe 填充排空与微批次梯度

## 1. 问题

实现填充排空事件表，以及保持整批均值梯度的微批次损失。

等时分段、同步更新的 CPU 调度模型与线性回归；不模拟通信、BatchNorm或重计算。

原文: [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) · Section 2.2, Algorithm; Figure 2, synchronous microbatch pipeline.

## 2. 核心公式

$$
F_{s,m}=s+m,\quad B_{s,m}=K+M-1+(K-1-s)+(M-1-m),\quad L=\frac{1}{N}\sum_c\sum_{i\in c}(x_i^T w-y_i)^2.
$$

## 3. 公式到代码

- `pipeline_schedule(stages: int, microbatches: int) -> list[tuple]` — Unit-duration forward/backward event records.
- `microbatch_mse(x: Tensor, y: Tensor, weight: Tensor, chunk_size: int) -> Tensor` — Differentiable scalar sample-mean squared error.

前向依赖来自左侧分段与同段前一微批次；反向从最后分段和最后微批次排空。累计每块平方误差之和，最后按样本数归约，不能平均各块均值。

## 4. 本题约定

stages、microbatches 和 chunk_size 为正整数。事件为 (phase, stage, microbatch, start)，phase 为 F/B；单事件占一个时隙。先完成所有前向，再按逆微批次顺序反向。x 为 (N,D)、y 为 (N,)、w 为 (D,)，允许最后一块较短；总损失除以 N。

## 5. 交互图

用 11 个确定性样本、4 维线性模型、学习率 0.08 的 35 步 SGD，对比整批和大小 4 的微批次轨迹；同时计算 4 分段在不同微批次数下的有效时隙率。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/gpipe --impl problems/ai_infra/gpipe/starter.py
paperleet test ai_infra/gpipe --impl problems/ai_infra/gpipe/solution.py
paperleet mutate ai_infra/gpipe
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
