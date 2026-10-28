# InstructBLIP 指令感知查询块 — 题解

第一层把指令信息写入 query，第二层用这些更新后的 query 检索图像。把指令仅送给后续语言模型会绕过本题机制。padding 仅屏蔽指令键；query 自身始终可见，保证每行 softmax 有定义。两个残差都属于本题明确的缩小块，不能省略；实际模型的投影、归一化与 FFN 留给扩展。

$$
U=Q+\operatorname{softmax}\left(Q[Q;I]^T/\sqrt D+M\right)[Q;I],\quad O=U+\operatorname{softmax}(UV^T/\sqrt D)V.
$$

参考实现按上述布局和归约逐步计算：

```python
def instruction_query_block(queries: Tensor, instruction: Tensor, visual: Tensor, instruction_mask: Tensor) -> Tensor:
    _floating(queries,instruction,visual)
    if queries.ndim!=2 or queries.numel()==0 or instruction.ndim!=2 or visual.ndim!=2 or visual.numel()==0 or instruction.shape[1]!=queries.shape[1] or visual.shape[1]!=queries.shape[1] or instruction_mask.shape!=(instruction.shape[0],) or instruction_mask.dtype!=torch.bool:
        raise ValueError("expected compatible (Q,D), (I,D), (N,D) features and boolean instruction mask")
    context=torch.cat((queries,instruction),0)
    valid=torch.cat((torch.ones(queries.shape[0],dtype=torch.bool,device=queries.device),instruction_mask),0)
    scores=queries@context.T/math.sqrt(queries.shape[1])
    updated=queries+torch.softmax(scores.masked_fill(~valid[None,:],float('-inf')),-1)@context
    return updated+torch.softmax(updated@visual.T/math.sqrt(queries.shape[1]),-1)@visual
```

错误实现检测覆盖这些常见问题：

- Removes instruction influence
- Lets padded instructions attend

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

恒等投影的两级查询自注意力和视觉交叉注意力，验证指令影响与 padding 隔离；不声称复现完整 Q-Former。

- [https://arxiv.org/abs/2305.06500](https://arxiv.org/abs/2305.06500)
