# ZeRO-2 分片 Adam 与状态内存

## 1. 问题

先平均梯度并按所有者分片，再更新局部 Adam 状态和汇总参数。

等长参数分片的单进程 ZeRO-2 语义与 mixed-precision Adam 状态字节模型；不包含激活和通信缓冲区。

原文: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) · Sections 5.1–5.3: optimizer, gradient and parameter partitioning.

## 2. 核心公式

$$
g_s=\frac1R\sum_r g^{(r)}_{sS:(s+1)S},\quad m_s=\beta_1m_s+(1-\beta_1)g_s,\quad v_s=\beta_2v_s+(1-\beta_2)g_s^2,\quad p_s\leftarrow p_s-\eta\frac{m_s/(1-\beta_1^t)}{\sqrt{v_s/(1-\beta_2^t)}+\epsilon}.
$$

## 3. 公式到代码

- `zero2_adam_step(params: Tensor, gradients: Tensor, first: Tensor, second: Tensor, step: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]` — New sharded parameters and Adam moments, without input mutation.
- `state_bytes(parameters: int, world_size: int, stage: int) -> int` — Per-rank model-state bytes for ZeRO stage 0, 1, 2 or 3.

归约维度是梯度副本的 rank，而不是参数所有者。归约后 reshape 成 (R,S) 即 reduce-scatter 的本地模拟；所有者独立更新，然后 flatten 表示 all-gather。stage 1/2/3 依次分片 12/2/2 字节类别。

## 4. 本题约定

参数与一、二阶状态均 (R,S)，梯度副本 (R,R*S)。返回 (新参数分片, 新一阶状态, 新二阶状态)。梯度跨数据并行 rank 取平均；step 从 1 开始，无 weight decay。内存模型假设每参数低精度权重 2、梯度 2、FP32 主权重与两个矩 12 字节；参数量须被 R 整除。

## 5. 交互图

两个 rank 分别处理确定性回归样本，以 0.04 学习率执行 45 步 Adam；对照不分片的独立 Adam，记录损失与状态字节。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/zero --impl problems/ai_infra/zero/starter.py
paperleet test ai_infra/zero --impl problems/ai_infra/zero/solution.py
paperleet mutate ai_infra/zero
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
