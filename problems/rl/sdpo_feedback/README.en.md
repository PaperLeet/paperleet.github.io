# 38. SDPO Feedback Selection and Self-Distillation

Select a successful self/peer solution or environment feedback following Table 2, then distill.

## Paper and scope

[Reinforcement Learning via Self-Distillation](https://arxiv.org/html/2601.20802v2) · Section 2, Algorithm 1, and teacher prompt Table 2 · [Code](https://github.com/lasgroup/SDPO)

Implement feedback routing and masked reverse KL; tiny programs and a categorical feedback teacher are supplied.

## Interface

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `select_feedback` · Required

Return `(solution_indices, feedback_available)` in that order, both shape `[N]` on the `question_ids` device (`N` attempts):

- `solution_indices`: `torch.long` row indices. A successful attempt selects itself; otherwise select the first successful attempt with the same question ID, or `-1` if absent.
- `feedback_available`: boolean Tensor, true when a solution exists or the attempt has rich feedback.

These routing outputs have no gradients. Rich feedback can make availability true while its solution index remains `-1`.

#### `feedback_reverse_kl` · Required

Return one floating scalar Tensor `loss`, shape `()`, matching student dtype/device. For `[N, L, V]` logits, sum reverse KL over vocabulary `V` and average tokens where `token_mask & feedback_available[:, None]` is true. Keep student gradients, freeze teacher logits. If no feedback tokens are eligible, return differentiable zero with zero student gradient.

### Formula and conventions

```python
select_feedback(question_ids, success, rich_feedback)
feedback_reverse_kl(student_logits, teacher_logits, token_mask, feedback_available)
```

`select_feedback` takes int64 `question_ids[B]`, bool `success[B]`, and bool `rich_feedback[B]`, B>0. Return `(solution_index[B], available[B])` in this order of priority:

1. If the original attempt succeeded, its solution index is **itself** and availability is True.
2. Otherwise choose a successful attempt on the **same question**, with availability True. If several exist, choose the smallest index.
3. With no successful solution, rich environment feedback gives index -1 and availability True.
4. With neither, return index -1 and availability False.

Successful solutions take precedence over rich feedback, following the paper's Table 2. Only the lowest-index tie-break is an exercise convention. `feedback_reverse_kl` takes finite student/teacher logits `[B,T,V]`, bool token mask `[B,T]`, and availability `[B]`. B and V are positive; T may be zero. Inputs are CPU float32/float64 with matching dtype/device. Return reverse KL averaged over all available tokens, or differentiable zero when none exist. Preserve inputs and allow gradients only through the student.

## Definition

The teacher additionally reads feedback f while scoring the original student prefixes. Loss is `sum_v p_student*(log p_student-log p_teacher)`. A position must satisfy both token_mask and feedback_available. A successful solution belongs in the teacher context; it does not replace the original student rollout and cannot come from another question. The teacher scores the original prefixes even when its context contains a different successful answer.

## Example

For questions=[7,8,7], success=[False,True,True], and no rich feedback: solution_index=[2,1,2], available=[True,True,True]. A failed singleton with rich feedback returns (-1,True); without feedback it returns (-1,False).

## Toy experiment

Evaluate six add-one, double, and square program attempts with supplied unit tests. Two successful attempts use their own solutions, two failed attempts use same-question successful solutions, and one failed attempt uses rich feedback. The final unavailable attempt must remain unchanged. A categorical teacher supplies logits; this is a teaching adaptation, not the full coding benchmark.

T0 checks contracts, T1 independent numerical examples, T2 gradients and invariants, and T3 the submitted functions in the supplied experiment.
