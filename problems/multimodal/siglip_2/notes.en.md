# SigLIP 2 masked patch self-distillation — Solution

Teacher inputs are already probabilities; applying another softmax changes their meaning. Stop their gradients, compute cross-entropy per patch, and divide the masked sum by the number of selected patches. Student gradients are zero outside the mask and (softmax(student/tau)-teacher)/(M*tau) inside it. This kernel isolates one auxiliary objective.

$$
L=-\frac{\sum_{b,p}m_{bp}\sum_k\operatorname{stopgrad}(q_{bpk})\log\operatorname{softmax}(z_{bp}/\tau)_k}{\sum_{b,p}m_{bp}}.
$$

The reference implementation follows the declared layout and reduction:

```python
def masked_patch_distillation(student: Tensor, teacher_prob: Tensor, mask: Tensor, temperature: float) -> Tensor:
    _floating(student, teacher_prob)
    if student.ndim != 3 or student.numel()==0 or teacher_prob.shape!=student.shape or mask.shape!=student.shape[:2] or mask.dtype!=torch.bool or not mask.any() or temperature<=0:
        raise ValueError("expected (B,P,K), nonempty boolean mask, and positive temperature")
    if (teacher_prob<0).any() or not torch.allclose(teacher_prob.sum(-1),torch.ones_like(teacher_prob.sum(-1)),atol=1e-6,rtol=1e-6):
        raise ValueError("teacher targets must be probabilities")
    per_patch=-(teacher_prob.detach()*F.log_softmax(student/temperature,dim=-1)).sum(-1)
    return (per_patch*mask.to(student.dtype)).sum()/mask.sum()
```

Common errors caught by mutation tests:

- Divides by all patches
- Backpropagates through teacher

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Implement the probability-distillation kernel of SigLIP 2 masked prediction with supplied teacher targets; exclude EMA encoders, augmentation, projection heads, and the full training recipe.

- [https://arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
