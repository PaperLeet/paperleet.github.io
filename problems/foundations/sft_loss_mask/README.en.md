# SFT Loss Mask: Train on Response Tokens

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Finite float32/64 logits [B,T,V], B,T,V>=1; int64 tokens [B,T] in [0,V); bool response_mask and attention_mask [B,T], all on CPU. Internally shift once: logits[:,t] predicts tokens[:,t+1]. A target j>=1 contributes iff response_mask[:,j] AND attention_mask[:,j]. Masks describe target positions, not predicting positions. Divide by the total contributing tokens across the batch; empty selection returns differentiable zero. Preserve logits dtype/device and gradients; masked predictions and final logits have zero gradients. Tokens are already valid IDs, not -100. Caller supplies response spans, including any desired EOS; no chat-template parsing, extra predecessor gate or mutation.

```python
sft_loss(logits: Tensor, tokens: Tensor, response_mask: Tensor, attention_mask: Tensor) -> Tensor
```

$$
L=\frac{\sum_{b,j\ge1}m_{b,j}\,\ell(z_{b,j-1},x_{b,j})}{\max(1,\sum_{b,j\ge1}m_{b,j})},\quad m=r\land a
$$

## Example

```python
sft_loss(torch.zeros(1, 3, 2), torch.tensor([[0, 1, 0]]),
         torch.tensor([[False, False, True]]), torch.ones(1, 3, dtype=torch.bool))
# tensor(0.693147...)
```

## Before implementing

Drop final logits and first targets/mask entries. Sum valid losses and divide by the total valid-token count, not a mean of per-sequence means. An empty selection must retain a zero-gradient graph.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
