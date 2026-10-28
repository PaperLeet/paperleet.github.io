# DistServe 分离队列与双延迟 SLO — 题解

## 推导与实现

分别维护prefill结束和decode结束时钟；decode只依赖自己的worker、当前请求的prefill完成和传输完成。用首末token之间的全部等待计算平均TPOT，不能只取裸decode服务时间。

$$
F_i=\max(A_i,F_{i-1})+P_i,\quad E_i=\max(F_i+K_i,E_{i-1})+(n_i-1)D_i,\quad TTFT_i=F_i-A_i,\quad TPOT_i=\frac{E_i-F_i}{n_i-1}.
$$

## 边界与梯度归属

输入为等长Python数值列表，arrivals非负且非递减；prefill/decode时间>0，transfer>=0，tokens为>=2整数。第一个token在prefill结束F产生，KV传输仅影响后续decode。两个worker分别保持FCFS输入顺序；传输可重叠。返回(first,last)；goodput=同时达标请求数/(max(last)-min(arrivals))，阈值含等号。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

三个同时到达的请求在prefill耗时4、decode耗时1、每请求3token下回放，扫描KV传输时间；比较双SLO达标goodput并记录first/last时刻。

一个prefill worker与一个decode worker的FCFS离散事件模型，包含KV传输等待；decode按请求串行，非实际批处理优化器。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| first_token_last_request | 12 |
| zero_transfer_goodput | 0.21428571 |
| long_transfer_goodput | 0 |

## 常见错误

- m1_colocated_clock: 让prefill错误等待decode完成
- m2_ignore_transfer: 忽略KV传输延迟

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · Sections 2 and 3: TTFT/TPOT, disaggregated prefill and decoding, goodput.
