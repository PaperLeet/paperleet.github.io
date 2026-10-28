# ZeRO-2 分片 Adam 与状态内存 — 题解

## 推导与实现

归约维度是梯度副本的 rank，而不是参数所有者。归约后 reshape 成 (R,S) 即 reduce-scatter 的本地模拟；所有者独立更新，然后 flatten 表示 all-gather。stage 1/2/3 依次分片 12/2/2 字节类别。

$$
g_s=\frac1R\sum_r g^{(r)}_{sS:(s+1)S},\quad m_s=\beta_1m_s+(1-\beta_1)g_s,\quad v_s=\beta_2v_s+(1-\beta_2)g_s^2,\quad p_s\leftarrow p_s-\eta\frac{m_s/(1-\beta_1^t)}{\sqrt{v_s/(1-\beta_2^t)}+\epsilon}.
$$

## 边界与梯度归属

参数与一、二阶状态均 (R,S)，梯度副本 (R,R*S)。返回 (新参数分片, 新一阶状态, 新二阶状态)。梯度跨数据并行 rank 取平均；step 从 1 开始，无 weight decay。内存模型假设每参数低精度权重 2、梯度 2、FP32 主权重与两个矩 12 字节；参数量须被 R 整除。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

两个 rank 分别处理确定性回归样本，以 0.04 学习率执行 45 步 Adam；对照不分片的独立 Adam，记录损失与状态字节。

等长参数分片的单进程 ZeRO-2 语义与 mixed-precision Adam 状态字节模型；不包含激活和通信缓冲区。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.38814764 |
| final_loss | 0.0031833261 |
| trajectory_error | 4.1633363e-16 |
| replicated_bytes | 16384 |
| stage2_bytes | 3840 |

## 常见错误

- m1_local_grad: 使用每rank自己的梯度而未做跨rank归约
- m2_missing_bias_correction: 遗漏Adam第一矩偏差校正

[ZeRO](https://arxiv.org/abs/1910.02054) · Sections 5.1–5.3: optimizer, gradient and parameter partitioning.
