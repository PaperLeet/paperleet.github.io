# InstructBLIP instruction-aware query block — Solution

The first stage writes instruction information into queries; the second uses updated queries to retrieve visual features. Sending instructions only to a later language model bypasses this mechanism. Padding masks instruction keys while queries remain visible, ensuring every softmax row is defined. Both residuals belong to the declared reduced block; full projections, normalization, and FFNs remain extensions.

$$
U=Q+\operatorname{softmax}\left(Q[Q;I]^T/\sqrt D+M\right)[Q;I],\quad O=U+\operatorname{softmax}(UV^T/\sqrt D)V.
$$

The reference implementation follows the declared layout and reduction:

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

Common errors caught by mutation tests:

- Removes instruction influence
- Lets padded instructions attend

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Two-stage query self-attention and visual cross-attention with identity projections, testing instruction influence and padding isolation; not a full Q-Former reproduction.

- [https://arxiv.org/abs/2305.06500](https://arxiv.org/abs/2305.06500)
