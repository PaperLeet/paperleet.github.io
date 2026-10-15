# 39. SDPO Top-K and Tail-Mass KL

Coarsen the vocabulary into student top-k entries and one tail bin while preserving total probability mass.

## Paper and scope

[Reinforcement Learning via Self-Distillation](https://arxiv.org/html/2601.20802v2) · Appendix A.3: Top-K KL Divergence · [Code](https://github.com/lasgroup/SDPO)

Implement top-k selection, complementary logsumexp, and reverse KL; tests and information curves are supplied.

## Interface

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `topk_tail_reverse_kl` · Required

Return one floating Tensor `divergence`, shape `student_logits.shape[:-1]`: remove **only the vocabulary axis** from `[..., V]`. Thus `[B, L, V]` returns `[B, L]`, and `[V]` returns a scalar `()`. Each value is reverse KL for student top-`k` tokens plus the complementary tail bin (`k=V` gives full KL). Preserve student dtype/device and gradients; freeze teacher logits. Do not apply batch/token mean reduction.

### Formula and conventions

```python
topk_tail_reverse_kl(student_logits, teacher_logits, k)
```

Inputs are same-shaped finite logits [...,V], V>0, with Python integer k in [1,V]. Use CPU float32/float64; return [...] with preserved dtype/device. Support noncontiguous inputs and stop teacher gradients. Select indices from the student and use the same indices for both distributions. At ties any torch.topk selection is valid. Indices need no gradient, but selected probabilities and tail mass do.

## Definition

Let I=TopK(p), with p_tail and q_tail the complementary masses. Return `sum_{i in I} p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)`. Omit the tail for k=V. Do not renormalize selected entries. Compute tail log mass with logsumexp over the complement to avoid cancellation in 1-sum(topk).

## Example

For p=[0.6,0.25,0.15], q=[0.3,0.2,0.5], k=1, the two bins are p=[0.6,0.4], q=[0.3,0.7], giving KL≈0.19204.

## Toy experiment

Increase k for a five-category distribution and plot coarse-grained KL approaching full KL. Full logits remain inputs to make the formula testable; this module does not implement distributed vocabulary communication or benchmark LLM memory savings.

T0 checks contracts, T1 independent numerical examples, T2 gradients and invariants, and T3 the submitted functions in the supplied experiment.
