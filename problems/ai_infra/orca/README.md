# Orca 迭代级动态批处理

## 1. 问题

让请求按到达顺序进入每轮空闲槽，完成即退出批次。

已完成prefill的离散生成迭代模拟，固定每轮一时间单位、每个活跃请求生成一token；不模拟selective batching算子。

原文: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) · Section 3, S1 and Figure 4: iteration-level scheduling; Section 4.2, Algorithm 1: FCFS admission.

## 2. 核心公式

$$
C_i=1+\max\{t:i\in B_t\},\qquad |B_t|\le K,\qquad \sum_t\mathbf{1}[i\in B_t]=L_i.
$$

## 3. 公式到代码

- `iteration_schedule(arrivals: list[int], lengths: list[int], capacity: int) -> tuple[list[tuple], list[int]]` — Per-iteration active request IDs and completion times in original order.

为每个请求维护剩余token。执行前接纳已到达且未开始的请求，执行后减少剩余量并立即释放完成请求。不能等整批最慢请求结束后才接纳新请求。

## 4. 本题约定

arrivals 与 lengths 为等长 Python 整数列表；到达>=0，长度>=1，capacity>=1。相同时刻按输入索引排序。每轮开头按 (arrival,index) 补满空位，已有请求不被抢占；完成时间为最后执行时隙右端。返回 (batches, completion_times)，空闲时隙保留空tuple。

## 5. 交互图

回放 6 个同时到达的异长请求，与请求级固定批次对照完成时间；扫描容量 1、2、3，比较平均延迟和占用槽。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/orca --impl problems/ai_infra/orca/starter.py
paperleet test ai_infra/orca --impl problems/ai_infra/orca/solution.py
paperleet mutate ai_infra/orca
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
