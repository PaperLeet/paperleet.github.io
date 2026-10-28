# Orca 迭代级动态批处理 — 题解

## 推导与实现

为每个请求维护剩余token。执行前接纳已到达且未开始的请求，执行后减少剩余量并立即释放完成请求。不能等整批最慢请求结束后才接纳新请求。

$$
C_i=1+\max\{t:i\in B_t\},\qquad |B_t|\le K,\qquad \sum_t\mathbf{1}[i\in B_t]=L_i.
$$

## 边界与梯度归属

arrivals 与 lengths 为等长 Python 整数列表；到达>=0，长度>=1，capacity>=1。相同时刻按输入索引排序。每轮开头按 (arrival,index) 补满空位，已有请求不被抢占；完成时间为最后执行时隙右端。返回 (batches, completion_times)，空闲时隙保留空tuple。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

回放 6 个同时到达的异长请求，与请求级固定批次对照完成时间；扫描容量 1、2、3，比较平均延迟和占用槽。

已完成prefill的离散生成迭代模拟，固定每轮一时间单位、每个活跃请求生成一token；不模拟selective batching算子。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| dynamic_mean_latency | 3.5 |
| static_mean_latency | 7 |
| tokens_executed | 11 |

## 常见错误

- m1_batch_barrier: 必须旧批全部结束才接纳请求
- m2_completion_start: 把完成时间记为最后时隙起点

[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) · Section 3, S1 and Figure 4: iteration-level scheduling; Section 4.2, Algorithm 1: FCFS admission.
