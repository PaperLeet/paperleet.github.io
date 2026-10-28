# Megatron 列并行与行并行 MLP — 题解

## 推导与实现

每个分片完整持有输入特征，仅拥有隐藏通道子集。GeLU 位于局部两次矩阵乘之间；第二层部分输出相加，不取平均。

$$
Y=\sum_{r=0}^{P-1}\operatorname{GeLU}(XA_r+a_r)B_r+b.
$$

## 边界与梯度归属

x:(N,D)，w1:(D,H)，b1:(H,)，w2:(H,O)，b2:(O,)；H 被 parts 整除。第一层按列、第二层按相同隐藏通道行分片。b2 在归约后仅加入一次。所有浮点输入保留梯度、dtype 与原始值。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

固定 9×3 输入和 8 个隐藏通道，以 2 个分片训练输出权重 30 步；逐步对比稠密实现的损失、权重和梯度。

单进程模拟两层 tensor-parallel MLP，包括偏置与反向梯度；采用 tanh GeLU 近似，不实现分布式通信。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| initial_loss | 0.0072715013 |
| final_loss | 0.00012874584 |
| trajectory_error | 5.5511151e-17 |

## 常见错误

- m1_average_reduce: 错误地平均各分片输出
- m2_bias_per_rank: 输出偏置按分片数重复加入

[Megatron-LM](https://arxiv.org/abs/1909.08053) · Section 3, Figure 3: column-partitioned first GEMM and row-partitioned second GEMM.
