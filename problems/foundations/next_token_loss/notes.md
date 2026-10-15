# 实现要点

先在每个序列内部错位，再归约。使用 logsumexp 或 log_softmax；最后一个位置的 logits 应有严格零梯度。

典型错误：

- 用当前位置 token 作为标签。
- 用后一位置 logits 预测前一 token。
- 把末位也计入平均分母。
- 切断损失梯度。

## 对齐与梯度

1. `logits[:, :-1]` 是产生预测的位置，`tokens[:, 1:]` 是被预测的标签。
2. 对每个预测行计算 `logsumexp(logits) - correct_logit`，再对全部目标求平均。
3. T=1 没有预测目标，返回与输入相连的零，避免空张量 mean 产生 NaN。

令 N=B(T−1)，参与损失的每行梯度为 `(softmax(logits) - one_hot(target))/N`。末位 logits 的梯度为零。先 flatten 再错位会把一个样本的末尾与下个样本的开头错误连接。

时间 O(BTV)，不需要构造完整 one-hot 标签。

先对 logits 减去每行最大值，再计算 logsumexp 与正确类别之差。否则 1e20 等共同偏移会吞掉 log(V)，连均匀分布的损失与梯度也会算错。参考解使用 float64 中间量，最终恢复输入 dtype。
