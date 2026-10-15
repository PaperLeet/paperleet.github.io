# GLM-5: Double-sided Off-policy Correction

Source: [GLM-5 technical report v2](https://arxiv.org/abs/2602.15763v2), §4.1.2 “Optimizing Asynchronous Training Stability”, Eqs. (3–5). Asynchronous rollouts can use different model versions. The report computes a current-to-rollout probability ratio and discards tokens whose ratios lie outside a double-sided interval.

## Your implementation

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `masked_is_weights` · Required

Return one **detached** floating Tensor `weights`, shape `[B, L]`, matching `logp` dtype/device (`B` responses, `L` tokens). Retained positions contain `exp(logp-rollout_logp)` strictly inside `(1-clip_low, 1+clip_high)`; padding, boundary and rejected positions are zero. Return the full token array, without reduction or input mutation.

#### `offpolicy_loss` · Required

Return one scalar floating Tensor `loss`, shape `()`, normalized by the number of **original valid tokens**, including ratio-rejected tokens. Gradient flows only through the direct current-`logp` factor; weights, rollout scores and advantages are fixed. An empty mask returns a graph-connected zero so `.backward()` remains valid; never return `.item()` or a detached scalar.

### Formula and conventions

Implement `masked_is_weights` and `offpolicy_loss`. Current and recorded rollout log-probabilities and the binary valid-token mask have shape `[B,L]`; per-response advantages have shape `[B]`. Each advantage is broadcast over its response's tokens.

$$r_{it}=\exp(\log\pi_{it}-\log\pi_{rollout,it}), \quad
w_{it}=\operatorname{sg}\!\left[r_{it}\mathbf1(1-\epsilon_l<r_{it}<1+\epsilon_h)\right],$$
$$L=-\frac{\sum_{it}m_{it}w_{it}\operatorname{sg}(A_i)\log\pi_{it}}{\max(1,\sum_{it}m_{it})}.$$

Here `sg` means stop-gradient. Return zero weights for padding and for **both endpoints**, not clamped endpoint values. The loss denominator counts **all original valid tokens**, including tokens rejected by the ratio test. Empty masks return a differentiable zero.

Freeze the ratio, rollout values and advantages. Only the direct current-log-probability factor receives the score-function gradient. Ignore arbitrary padded values before arithmetic, including NaNs, and safely discard extremely large rejected ratios. Compare the computed ratio against bounds in its native dtype. Detached exponentials may overflow internally; select zero with `torch.where` rather than multiply an infinity by zero. Preserve float32/float64 dtype and do not mutate inputs. Shape mismatches and invalid widths must raise `ValueError`; require `0 < clip_low < 1` and finite `clip_high > 0`. Unmasked log-probabilities and advantages are finite; the mask contains only 0/1 or booleans.

## Paper versus implementation conventions

The report's Eq. (5) uses a **strict interval**. Current official Slime's `importance_weights(mode="mis")` includes endpoints. This exercise deliberately follows the report's strict boundaries; it does not claim bitwise equivalence to that helper.

The report's displayed objective does not explicitly annotate stop-gradient. We specify detached correction weights, consistent with Slime's REINFORCE path supplying detached current probabilities to its correction callback. The denominator and empty-mask behavior above are explicit exercise contracts. The default widths `.2/.2` are classroom parameters, not a claim about the report's production hyperparameters.

Official Slime is pinned to commit `05e3b017fed9e02de418c2fbac9d75602a534be5`: [importance weights](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/utils/ppo_utils.py#L25) and [policy loss](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/backends/megatron_utils/loss.py#L923) (`icepop_function`, plus detached `train_log_probs_for_tis` in `policy_loss_function`). Their helper names are not a claim that the complete GLM training recipe is ICE-POP.

## Toy and verification

`experiment.py` supplies a ratio sweep and a deterministic two-action bandit. The sweep plots your weights, a clamped comparison, and actual gradients. The bandit freezes a rollout snapshot for five updates before refreshing it, making the correction operate on stale data. This is a **PaperLeet teaching adaptation**, not a toy from the authors or a reproduction of coding-agent results. T3 artifacts come from your submitted functions; static Lab previews come from the reference.

T0 checks empty masks/contracts; T1 uses a scalar-loop oracle; T2 checks strict endpoints, gradient detachment, padding, numerical extremes and token averaging; T3 executes the supplied toy. No model download, paid API, async infrastructure, tokenizer gateway or coding sandbox is required.
