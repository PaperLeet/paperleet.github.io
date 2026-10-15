# GSPO Sequence Ratios and Clipping

Implement the sequence importance ratio and clipped policy loss.

Paper: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v1). Anchors: Section 4.1, Eqs. (6–8): sequence objective and geometric importance ratio; Section 4.2, Eq. (11): equal token gradient weights.

## Selected core

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `sequence_ratio` · Required

Return one floating Tensor `ratio`, shape `[B]`, for `[B, L]` scores/mask (`B` responses, `L` token positions). Each value is the geometric mean of valid-token importance ratios; an empty response returns `1`. Preserve input floating dtype/device and gradients to `logp`; `old_logp` is fixed. Do not reduce over the batch.

#### `gspo_loss` · Required

Return one scalar floating Tensor `loss`, shape `()`, the negative mean clipped sequence objective over **all `B` responses**, including zero contributions from empty responses. Retain `logp` gradients, freezing `old_logp` and advantages. An all-empty mask returns a differentiable zero, not a Python number.

### Formula and conventions

Two functions; group advantages, sampling and distributed RL are outside this module.

$$
s_i=\exp\left(\frac{\sum_t m_{it}(\log\pi_{it}-\log\pi^{old}_{it})}{\sum_t m_{it}}\right)
$$

- `sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor`
- `gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2) -> Tensor`

## Contract and experiment

`logp`, `old_logp`, and binary `mask` have shape `[B,L]`; `advantages` has shape `[B]`. First sum the valid log-ratios and divide by valid length, then exponentiate. Clip that single ratio per response, multiply by the detached advantage, take the smaller surrogate, and negate the batch mean.

The old policy and advantages are frozen. A response with no valid tokens has ratio 1 and contributes zero to the batch mean. Padding must be neutralized **before** exponentiation, including extreme padded scores. Duplicating tokens within a response leaves the loss unchanged. This module implements GSPO, not GSPO-token. The default `.2` clip is an inspectable teaching setting; it is not a claim about Qwen's production hyperparameter.

Try token ratios `[2, 0.5]`: their geometric ratio is 1, so both valid log-probabilities receive the same gradient. A positive-advantage response with ratio 1.5 clips; a negative-advantage response with ratio 0.5 also clips. The toy saves the submitted implementation's ratios and gradient matrix.

## Toy provenance

**PaperLeet teaching adaptation.** Analytic three-response diagnostic written for PaperLeet; not Qwen training results. The test harness calls your submitted functions and records the resulting metrics and curves. The Lab's static preview comes from the reference; your Run/Submit artifacts come from your code. The experiment and test scaffolding are supplied in `experiment.py` and `tests/`.

## Sources and verification

[Author explanation (no author training-code release verified)](https://qwenlm.github.io/blog/gspo/). All exercise implementations and synthetic data are written for PaperLeet. The paper link pins the version used for the formulas.

T0 checks contracts, T1 compares with the reference, T2 checks mathematical and gradient properties, and T3 runs the toy through your implementation. Implement the starter functions in the online editor, then Run or Submit. The official large-model training system is not required.
