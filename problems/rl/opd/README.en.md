# On-policy Distillation at Student Prefixes

Sample from the student and distill full-vocabulary feedback at the same prefixes.

Paper: [On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649v3). Anchors: Section 3, Algorithm 1: GKD and student data fraction lambda; Section 3, Eqs. (2–4): token divergences on student-generated prefixes; Appendix A.7: forward/reverse KL interpretation (separate conceptual example).

## Selected core

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `sample_student` · Required

Return `(tokens, prefixes)`, a tuple of two `torch.long` Tensors of shape `[batch_size, length]`, on `transition_logits.device`, with no gradients. `tokens` contains sampled IDs in `[0, V-1]`; `prefixes[:, 0]` is the BOS ID `V` and `prefixes[:, 1:] == tokens[:, :-1]`. Sampling uses the supplied generator. Do not return probabilities or log-probabilities.

#### `distillation_loss` · Required

Return one floating scalar Tensor `loss`, shape `()`, for logits `[B, L, V]`. Sum KL across the vocabulary `V`, then average only valid tokens selected by `[B, L]` mask. `reverse` means KL(student || teacher); `forward` means KL(teacher || student). Keep student gradients, freeze teacher logits; empty masks produce graph-connected zero. Preserve the input floating dtype/device.

### Formula and conventions

GKD with lambda=1 and explicit forward/reverse KL; a supplied three-token Markov LM replaces large language models.

$$
L=\mathbb E_{y\sim p_S}[\frac1{|y|}\sum_t D_{KL}(p_S(\cdot|y_{<t})\Vert p_T(\cdot|y_{<t}))]
$$

- `sample_student(transition_logits: Tensor, batch_size: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor]`
- `distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, direction: str='reverse') -> Tensor`

## Contract and experiment

Implement `sample_student` and `distillation_loss`. The sampling table has shape `[V+1,V]`; row `V` is BOS and row `k` is the distribution after token `k`. Return `(tokens, prefixes)`, both `[B,L]` integer tensors. The first prefix is BOS; every subsequent prefix is the previous sampled token. Use the supplied generator and categorical sampling at temperature 1, without a gradient through discrete sampling. Sequences have fixed length and no EOS in this toy.

The loss inputs are `[B,L,V]` logits evaluated for **both models at those same student-generated prefixes**, plus `[B,L]` binary mask. `reverse` means KL(student || teacher); `forward` means KL(teacher || student). Sum over the entire vocabulary, average only valid tokens, freeze the teacher, and use log-softmax. All-padding gives differentiable zero. Padded logits may be nonfinite and must be neutralized before log-softmax. Valid logits are finite.

This is the lambda=1 GKD variant. It is not sampled-token OPD, a policy-gradient estimator, or teacher-generated training data. The supplied training loop learns four conditional distributions with a three-token model. Its evaluation metric is mean forward KL over all four states, while the training objective is reverse KL on fresh student prefixes.

## Toy provenance

**PaperLeet teaching adaptation.** PaperLeet fixed-length categorical Markov model, not the original T5 or Gaussian-mixture experiment. The test harness calls your submitted functions and records the resulting metrics and curves. The Lab's static preview comes from the reference; your Run/Submit artifacts come from your code. The experiment and test scaffolding are supplied in `experiment.py` and `tests/`.

## Sources and verification

[TRL implementation (independent of the paper authors; not the reference for this exercise)](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py). All exercise implementations and synthetic data are written for PaperLeet. The paper link pins the version used for the formulas.

T0 checks contracts, T1 compares with the reference, T2 checks mathematical and gradient properties, and T3 runs the toy through your implementation. Implement the starter functions in the online editor, then Run or Submit. The official large-model training system is not required.
