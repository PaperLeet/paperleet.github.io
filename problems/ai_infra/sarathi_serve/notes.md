# Sarathi-Serve Decode 优先与分块 Prefill — 题解

## 推导与实现

先扣除所有decode的一个token，再依次处理prefill。若长prompt用完本轮块长但尚未完成，它保留队首位置。不要为新prompt挤掉已有decode，也不要一次执行完整长prompt。

$$
B_{prefill}=B-|D|,\qquad c_i=\min(r_i,C,B_{remaining}),\qquad |D|+\sum_i c_i\le B.
$$

## 边界与梯度归属

decodes 是唯一非负请求ID列表；prefills 是按FIFO排列的正剩余token数。budget>=len(decodes)，chunk_size>=1。每个decode本轮运行一次。队首prefill消耗 min(剩余,chunk_size,预算)；未完成则停止接纳后续prefill，本轮每个请求最多一块。返回dict含decodes、prefills(index,count)、remaining(含零完成项)。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

固定两个活动decode，prefill长度 13、3、9，预算 8、最大块 4；循环消费队列，验证每轮decode不被暂停及总prefill工作守恒。

一次混合批次的预算分配和多轮回放；以固定token预算与最大块长替代论文的性能模型。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| prefill_tokens | 25 |
| decode_stalls | 0 |
| maximum_batch_tokens | 8 |
| maximum_chunk | 4 |

## 常见错误

- m1_no_decode_reservation: prefill误用decode保留的预算
- m2_unchunked_prefill: 忽略prefill块大小上限

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) · Section 4.1–4.2; Algorithm 3, stall-free batching.
