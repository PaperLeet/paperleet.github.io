# DAPO Dynamic Filtering and Token Objective

Retain informative groups and implement asymmetric token-normalized clipping.

Paper: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476v1). Anchors: Section 3, Eqs. (8–11): token normalization, asymmetric clipping and dynamic sampling; Sections 3.1–3.3; Section 4.1 clip_low=.2 and clip_high=.28.

## Selected core

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `dynamic_group_mask` · Required

Return one boolean Tensor `keep`, shape `[P]` on the rewards device, for rewards `[P, G]` (`P` prompts, `G` group members). `True` keeps a prompt whose rewards contain both outcomes; uniform-success and uniform-failure groups are `False`. This is one decision per prompt, not per response/token.

#### `dapo_loss` · Required

Return one scalar floating Tensor `loss`, shape `()`: negative clipped objective sum divided by the total number of retained valid tokens across `[N, L]` inputs. Retain current `logp` gradients and freeze `old_logp`/`advantages`; padding has zero gradient. An empty mask returns a differentiable zero Tensor. Do not return per-response losses or diagnostics.

### Formula and conventions

Two bounded core functions. Oversampling orchestration, overlong reward shaping, and distributed training are not reproduced.

$$
L=-\frac{\sum_{i,t}m_{it}\min(r_{it}A_i,\mathrm{clip}(r_{it},1-\epsilon_l,1+\epsilon_h)A_i)}{\sum_{i,t}m_{it}}
$$

- `dynamic_group_mask(rewards: Tensor) -> Tensor`
- `dapo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_low: float=0.2, clip_high: float=0.28) -> Tensor`

## Contract and experiment

`dynamic_group_mask` receives binary outcome rewards `[prompts,group]` with group size at least 2. Return a boolean `[prompts]` vector retaining groups with at least one success and one failure. Both `0/1` and `-1/1` reward encodings are supported. Selecting mixed groups is the core filtering primitive; a production sampler would replenish the batch after filtering.

`dapo_loss` accepts `[B,L]` current/old scores and binary masks plus `[B]` group-normalized advantages. These responses have already been retained by the caller. Freeze old scores and advantages, use lower/upper clip widths independently, and normalize by the **total valid token count**, not by each response's length. Defaults are `.2/.28`. There is no reference-policy KL term. An empty mask returns differentiable zero; mask extreme padded scores before exponentiation.

The toy keeps only the mixed group among all-fail, mixed and all-success examples, then scans ratios from .5 to 1.5. A ratio 1.25 for positive advantage should still have nonzero gradient under clip_high=.28. The tests also distinguish unequal response-length weighting.

## Toy provenance

**PaperLeet teaching adaptation.** PaperLeet synthetic binary-reward groups and ratio sweep; no AIME reproduction. The test harness calls your submitted functions and records the resulting metrics and curves. The Lab's static preview comes from the reference; your Run/Submit artifacts come from your code. The experiment and test scaffolding are supplied in `experiment.py` and `tests/`.

## Sources and verification

[Official framework linked by the paper (redirects to verl-project/verl)](https://github.com/volcengine/verl). All exercise implementations and synthetic data are written for PaperLeet. The paper link pins the version used for the formulas.

T0 checks contracts, T1 compares with the reference, T2 checks mathematical and gradient properties, and T3 runs the toy through your implementation. Implement the starter functions in the online editor, then Run or Submit. The official large-model training system is not required.
