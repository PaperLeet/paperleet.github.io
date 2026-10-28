# BLIP momentum soft-target mixing — Solution

Normalize teacher logits across all queue columns, then mix the soft distribution with one-hot positives. Each target row still sums to one, giving student gradient (softmax(logits)-targets)/B. Alpha zero recovers hard labels; alpha one uses only distillation. Momentum updates belong outside this loss, so teacher gradients must stop.

$$
q_{ij}=\alpha\,\operatorname{stopgrad}(\operatorname{softmax}(s^m_i)_j)+(1-\alpha)\mathbf 1[j=p_i],\quad L=-\frac1B\sum_{i,j}q_{ij}\log\operatorname{softmax}(s_i)_j.
$$

The reference implementation follows the declared layout and reduction:

```python
def momentum_target_loss(logits: Tensor, teacher_logits: Tensor, positives: Tensor, alpha: float) -> Tensor:
    _floating(logits, teacher_logits)
    if logits.ndim!=2 or logits.numel()==0 or logits.shape!=teacher_logits.shape or positives.shape!=(logits.shape[0],) or positives.dtype!=torch.long or not 0<=alpha<=1:
        raise ValueError("expected (B,N) logits, (B,) long positives and alpha in [0,1]")
    if (positives<0).any() or (positives>=logits.shape[1]).any(): raise ValueError("positive index out of range")
    hard=(torch.arange(logits.shape[1],device=logits.device)[None,:]==positives[:,None]).to(logits.dtype)
    targets=alpha*torch.softmax(teacher_logits.detach(),-1)+(1-alpha)*hard
    return -(targets*F.log_softmax(logits,-1)).sum(-1).mean()
```

Common errors caught by mutation tests:

- Reverses teacher and hard-label weights
- Updates momentum teacher through targets

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Implement BLIP momentum soft-target distillation for one retrieval direction with rectangular negative queues; omit queue updates, momentum encoders, and CapFilt data generation.

- [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
