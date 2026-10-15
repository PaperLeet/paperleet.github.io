# SFT Loss Mask：只学习回答 token

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

logits 为有限 float32/64 [B,T,V]，B,T,V≥1；tokens 为 int64 [B,T] 且在 [0,V)；response_mask、attention_mask 为 bool [B,T]，均在 CPU。本函数内部只错位一次：logits[:,t] 预测 tokens[:,t+1]。目标位置 j≥1 仅在 response_mask[:,j] 与 attention_mask[:,j] 同时为真时参与；掩码标记目标位置，不是产生预测的位置。对全 batch 有效目标 token 总数求平均；无有效目标时返回可反传的零。保持 logits dtype/device 和梯度；屏蔽预测位置及末位梯度为零。token 均为合法编号，不传 -100。回答区间及 EOS 是否参与由调用者给定；不解析聊天模板、不额外检查前一位置掩码、不修改输入。

```python
sft_loss(logits: Tensor, tokens: Tensor, response_mask: Tensor, attention_mask: Tensor) -> Tensor
```

$$
L=\frac{\sum_{b,j\ge1}m_{b,j}\,\ell(z_{b,j-1},x_{b,j})}{\max(1,\sum_{b,j\ge1}m_{b,j})},\quad m=r\land a
$$

## 示例

```python
sft_loss(torch.zeros(1, 3, 2), torch.tensor([[0, 1, 0]]),
         torch.tensor([[False, False, True]]), torch.ones(1, 3, dtype=torch.bool))
# tensor(0.693147...)
```

## 动手前想一想

logits 去尾，标签和两个掩码去头。先求有效 token 的损失总和，再除以有效数量，不能先按序列平均。全屏蔽时仍需保留零梯度计算图。

## 测试与实验

T0 检查接口；T1 对齐独立基准；T2 检查边界、不变量、适用梯度和输入保护；T3 在配套实验中调用你的函数。图表来自实际运行，不声称大模型训练效果或吞吐提升。

## 参考资料

- [Primary documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
