# 36. OPSD Dual-Context Rollout Alignment

Align next-token logits for the same student response under unequal context lengths.

## Paper and scope

[Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/html/2601.18734v3) · Section 3.2 and Algorithm 1 · [Code](https://github.com/siyan-zhao/OPSD)

Implement causal position selection, variable-length masking, and teacher detachment; logits and a toy training loop are supplied.

## Interface


### Return requirements

#### `align_rollout_logits` — Required

Return the Python tuple `(student, teacher, mask)` in that order. The floating student/teacher tensors are `[B,T,V]`, with batch size B, vocabulary size V and T=max(response_lengths); each preserves its source logits' dtype/device. mask is a boolean Tensor `[B,T]` on the same device, True for valid response positions. Padded logits are exactly zero; all-empty responses yield `[B,0,V]`, `[B,0,V]`, `[B,0]`. Keep gradients only to selected student positions, detach teacher values, and do not mutate inputs.

### Formula and conventions
```python
align_rollout_logits(student_logits, teacher_logits, student_prompt_lengths, teacher_prompt_lengths, response_lengths)
```

`student_logits[B,Ls,V]` and `teacher_logits[B,Lt,V]` are right-padded model outputs. Both contexts append the **same student rollout**; only the teacher context contains the reference solution. Prompt lengths are positive int64 `[B]`; response lengths are nonnegative int64 `[B]`. Inputs are CPU tensors, batch size is positive, logits are finite float32/float64, and each physical sequence fits its prompt and response. Return `(student[B,T,V], teacher[B,T,V], mask[B,T])`, where `T=max(response_lengths)`. Zero padded output positions. Detach teacher values and retain student gradients.

## Definition

The prediction for response token j is at `prompt_length - 1 + j`, before that token is consumed. Use a separate prompt length for each branch.

## Example

With student prompt length 2, teacher prompt length 5, and response length 3, select positions `[1,2,3]` and `[4,5,6]`; mask is `[True,True,True]`.

## Toy experiment

Train categorical logits on three unequal-length responses and track correct-token probability at the actual response positions. This teaching experiment supplies privileged targets and tests alignment; it does not run an LLM or reproduce reasoning benchmarks.

T0 checks contracts, T1 independent numerical examples, T2 gradients and invariants, and T3 the submitted functions in the supplied experiment.
