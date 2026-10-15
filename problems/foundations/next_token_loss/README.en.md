# Next-token Loss: Shifted Causal Targets

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Finite float32/64 logits [B,T,V], B,T,V>=1; int64 tokens [B,T] in [0,V), same CPU device. Logits at t predict tokens at t+1. Return scalar NLL averaged over B*(T-1) targets, preserving logits dtype/device and gradients. Ignore the last logit and first target; never cross sequence boundaries. T=1 returns a differentiable zero with zero logits gradients. No padding, masks, smoothing or mutation.

```python
next_token_loss(logits: Tensor, tokens: Tensor) -> Tensor
```

$$
L=\frac{1}{B(T-1)}\sum_{b,t<T-1}\left[\operatorname{logsumexp}(z_{b,t})-z_{b,t,x_{b,t+1}}\right]
$$

## Example

```python
next_token_loss(torch.zeros(1, 3, 2), torch.tensor([[0, 1, 0]]))
# tensor(0.693147...)
```

## Before implementing

Shift within each sequence before reduction. Use logsumexp or log_softmax. The final logit position must receive exactly zero gradient.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling)
