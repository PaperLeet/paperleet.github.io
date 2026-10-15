# 37. OPSD Pointwise Divergence Clipping

Upper-clip vocabulary contributions to forward and reverse KL.

## Paper and scope

[Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/html/2601.18734v3) · Section 3.2: Per-Token Pointwise Divergence Clipping · [Code](https://github.com/siyan-zhao/OPSD)

Implement signed divergence terms, pointwise clipping, and response-level averaging; validation and experiments are supplied.

## Interface


### Return requirements

#### `clipped_distillation_loss` — Required

Return one zero-dimensional floating Tensor (`shape == ()`) with the logits' dtype/device, not a Python float or per-token loss. Sum clipped signed vocabulary contributions per token, average valid tokens separately in each response, then average all B responses; B is batch size. Empty responses contribute zero and remain in the batch denominator. The loss may be negative. Preserve student-only gradients, with differentiable zero for all-masked inputs or zero-length responses (`T=0`, `B>0`); detach teacher logits and do not mutate inputs.

#### `_validate` — Provided helper

Return None when tau is finite and positive and direction is 'forward' or 'reverse'. Invalid arguments raise ValueError; no loss or transformed arguments are returned.

### Formula and conventions
```python
clipped_distillation_loss(student_logits, teacher_logits, mask, tau, direction='forward')
```

Inputs are same-shaped finite logits `[B,T,V]`, boolean mask `[B,T]`, finite positive scalar tau, and direction=`forward`/`reverse`. B and V are positive; T may be zero. Use CPU float32/float64. Return a scalar loss with preserved dtype/device and student-only gradients. Average valid tokens within each response, then average all B responses; an empty response contributes zero. Do not mutate inputs.

## Definition

Let p=softmax(student), q=stopgrad(softmax(teacher)). Forward terms are `q*(log q-log p)`; reverse terms are `p*(log p-log q)`. Apply `min(term,tau)` to **each vocabulary contribution**, then sum over vocabulary. Retain negative contributions: the clipped objective can be negative. Do not clamp at zero or clip the summed KL.

## Example

For p=[0.03,0.17,0.8], q=[0.8,0.15,0.05], tau=0.01, forward clipped loss is approximately -0.14741. This negative value is allowed.

## Toy experiment

Sweep tau on three-category distributions and compare pointwise clipping with aggregate clipping. This is a teaching adaptation of the formula, not the Qwen training experiment.

T0 checks contracts, T1 independent numerical examples, T2 gradients and invariants, and T3 the submitted functions in the supplied experiment.
