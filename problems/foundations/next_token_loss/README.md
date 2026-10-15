# Next-token Loss：下一词预测的错位对齐

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

logits 为有限 float32/64 [B,T,V]，B,T,V≥1；tokens 为同设备 int64 [B,T]，编号在 [0,V)。位置 t 的 logits 预测 t+1 的 token；对 B×(T−1) 个目标求平均，返回同 dtype/device 的标量并保留 logits 梯度。末位 logits 不参与，首位 token 不是目标；不得跨样本拼接。T=1 返回可反传的零，所有 logits 梯度为零。无 padding、掩码和平滑，不修改输入。

```python
next_token_loss(logits: Tensor, tokens: Tensor) -> Tensor
```

$$
L=\frac{1}{B(T-1)}\sum_{b,t<T-1}\left[\operatorname{logsumexp}(z_{b,t})-z_{b,t,x_{b,t+1}}\right]
$$

## 示例

```python
next_token_loss(torch.zeros(1, 3, 2), torch.tensor([[0, 1, 0]]))
# tensor(0.693147...)
```

## 动手前想一想

先在每个序列内部错位，再归约。使用 logsumexp 或 log_softmax；最后一个位置的 logits 应有严格零梯度。

## 测试与实验

T0 检查接口；T1 对齐独立基准；T2 检查边界、不变量、适用梯度和输入保护；T3 在配套实验中调用你的函数。图表来自实际运行，不声称大模型训练效果或吞吐提升。

## 参考资料

- [Primary documentation](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling)
