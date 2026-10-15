# 39. SDPO Top-K 与尾部概率散度

将词表压缩成学生 top-k 项和一个尾部桶，保留分布的总概率质量。

## 论文与范围

[Reinforcement Learning via Self-Distillation](https://arxiv.org/html/2601.20802v2) · Appendix A.3: Top-K KL Divergence · [Code](https://github.com/lasgroup/SDPO)

实现 top-k 选择、尾部 logsumexp 和 reverse KL；测试与信息损失曲线已提供。

## 接口

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `topk_tail_reverse_kl` · 需要实现

返回浮点 Tensor `divergence`，形状为 `student_logits.shape[:-1]`，从 `[..., V]` 中 **只移除词表轴**。因此 `[B, L, V]` 返回 `[B, L]`，`[V]` 返回标量 `()`。每项是 student top-`k` token 与剩余 tail bin 上的反向 KL（`k=V` 为完整 KL）。保留 student 的 dtype/device 和梯度，teacher logits 固定；不对 batch/token 求平均。

### 公式与约定

```python
topk_tail_reverse_kl(student_logits, teacher_logits, k)
```

输入同形有限 logits [...,V]，V>0、k 为 [1,V] 中的 Python 整数。CPU float32/float64。输出 [...]，保持 dtype/device。支持非连续输入；教师不接收梯度。top-k 索引来自学生，两边使用完全相同的索引；并列值可使用 torch.topk 的任意确定结果。不要求对排序索引求梯度，但选中概率和尾部质量都要可微。

## 定义

令 I=TopK(p)，p_tail=sum_{v不在I}p_v，q_tail 同理。输出 `sum_{i in I} p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)`。k=V 时省略尾部项。不能重新归一化 top-k 项；用补集 logsumexp 计算尾部 log mass，避免 1-sum(topk) 的消减误差。

## 例子

p=[0.6,0.25,0.15]，q=[0.3,0.2,0.5]，k=1：两桶为 p=[0.6,0.4]、q=[0.3,0.7]，KL≈0.19204。

## 小实验

五分类分布中逐步增加 k，画出粗粒度 KL 接近完整 KL 的曲线。本题仍接收完整 logits 来测试公式，未实现分布式词表通信或验证大模型显存收益。

T0 检查接口；T1 使用独立数值例子；T2 检查梯度与不变量；T3 在小实验中实际调用提交实现。
