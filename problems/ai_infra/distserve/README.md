# DistServe 分离队列与双延迟 SLO

## 1. 问题

回放独立prefill/decode服务队列，并按TTFT与TPOT共同约束计算goodput。

一个prefill worker与一个decode worker的FCFS离散事件模型，包含KV传输等待；decode按请求串行，非实际批处理优化器。

原文: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · Sections 2 and 3: TTFT/TPOT, disaggregated prefill and decoding, goodput.

## 2. 核心公式

$$
F_i=\max(A_i,F_{i-1})+P_i,\quad E_i=\max(F_i+K_i,E_{i-1})+(n_i-1)D_i,\quad TTFT_i=F_i-A_i,\quad TPOT_i=\frac{E_i-F_i}{n_i-1}.
$$

## 3. 公式到代码

- `disaggregated_trace(arrivals: list[float], prefill: list[float], decode: list[float], transfer: list[float], tokens: list[int]) -> tuple[list[float], list[float]]` — First- and last-token times under independent FCFS workers.
- `slo_goodput(arrivals: list[float], first: list[float], last: list[float], tokens: list[int], ttft_limit: float, tpot_limit: float) -> float` — Joint TTFT/TPOT-compliant request rate over the observation horizon.

分别维护prefill结束和decode结束时钟；decode只依赖自己的worker、当前请求的prefill完成和传输完成。用首末token之间的全部等待计算平均TPOT，不能只取裸decode服务时间。

## 4. 本题约定

输入为等长Python数值列表，arrivals非负且非递减；prefill/decode时间>0，transfer>=0，tokens为>=2整数。第一个token在prefill结束F产生，KV传输仅影响后续decode。两个worker分别保持FCFS输入顺序；传输可重叠。返回(first,last)；goodput=同时达标请求数/(max(last)-min(arrivals))，阈值含等号。

## 5. 交互图

三个同时到达的请求在prefill耗时4、decode耗时1、每请求3token下回放，扫描KV传输时间；比较双SLO达标goodput并记录first/last时刻。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/distserve --impl problems/ai_infra/distserve/starter.py
paperleet test ai_infra/distserve --impl problems/ai_infra/distserve/solution.py
paperleet mutate ai_infra/distserve
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
