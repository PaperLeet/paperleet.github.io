# GRPO Group-Relative Policy Objective

## 1. What the paper solves

GRPO samples a group of responses for each prompt and uses relative rewards in place of a value model, reducing the critic cost of reinforcement learning for language models. [DeepSeekMath Section 4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) presents the objective and group estimator.

Module goal: Implement within-group advantages and the clipped policy objective averaged over responses. Implement the advantage and loss functions; masking helpers, diagnostics, and sampling experiments are supplied. Completing this exercise completes one selected core module of the paper.

## 2. Core equations and task

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `grpo_advantages` · Required

Return one floating Tensor `advantages`, shape `[B * G]`, in the original adjacent-group order (`B` prompts, `G = group_size` responses each). Preserve `rewards` dtype/device. Center within each group; divide by sample std plus `eps` only when `scale_std=True`. Do not return a `[B, G]` matrix or a scalar reduction.

#### `grpo_loss` · Required

Return `(loss, stats)`, a two-element tuple containing a scalar Tensor and a dictionary. `loss` has shape `()`: average valid tokens within each response, then average all `N` responses for inputs `[N, L]` (`L` token positions). Empty responses contribute zero. Retain gradients to `logp`; freeze old/reference scores and advantages. The dictionary has exactly the documented fields `kl_mean`, `clipfrac`, `ratio_mean`: floating scalar Tensors of shape `()`, detached, averaged over all valid tokens. `clipfrac` counts `abs(ratio - 1) > clip_eps`. With no valid tokens, all three are zero.

#### `masked_logprobs` · Provided

Returns `(masked_logp, masked_old_logp, masked_ref_logp)` in the same order as the three score inputs. Each Tensor keeps its input's `[N, L]` shape, dtype and device, with zero at masked positions and unchanged scores elsewhere. This helper preserves active-position gradient paths; the loss caller freezes old/reference scores separately.

#### `token_statistics` · Provided

Returns a `dict[str, Tensor]` for `[N, L]` token arrays. The dictionary has exactly the documented fields `kl_mean`, `clipfrac`, `ratio_mean`: floating scalar Tensors of shape `()`, detached, averaged over all valid tokens. `clipfrac` counts `abs(ratio - 1) > clip_eps`. With no valid tokens, all three are zero.

### Formula and conventions

Implement `grpo_advantages` and `grpo_loss`. Rewards have shape `[B*G]`, with each prompt's $G$ responses stored contiguously. Require $G\ge2$ and a reward count divisible by $G$. Using group mean $\bar r_g$ and sample standard deviation $s_g$:

$$
\hat A_i=\frac{r_i-\bar r_g}{s_g+\varepsilon},\qquad
\rho_{i,t}=e^{\log\pi_{i,t}-\log\pi^{old}_{i,t}},\quad
k3_{i,t}=e^{d_{i,t}}-d_{i,t}-1,
\quad d=\log\pi^{ref}-\log\pi.
$$

With `scale_std=False`, subtract only the mean. Defaults are $\varepsilon=10^{-4}$ and standard-deviation `correction=1`. The three log-probability tensors and mask have shape `[N,L]`, where $N=BG$. Advantages have shape `[N]`; add a final dimension before broadcasting.

$$
L=-\frac1N\sum_i\frac1{\sum_t m_{i,t}}\sum_t m_{i,t}
\left[\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-c,1+c)\hat A_i)-\beta k3_{i,t}\right].
$$

Defaults are $c=0.2$ and $\beta=0.04$. An all-padding response contributes zero, but every row still counts in the response average. Return a scalar loss and `kl_mean, clipfrac, ratio_mean`. These diagnostics average over valid tokens and are detached; clipfrac counts $|\rho-1|>c$.

## 3. From equations to code

```text
rewards [B*G] → reshape [B,G] → subtract group mean
Optionally divide by group standard deviation → flatten
logp/old/ref [N,L] → detach old/ref → compute ratios and k3
advantages [N] → [N,1] → multiply by each token's ratio
Mask the pessimistic token objective
Sum each row / valid row length → average over responses
```

The starter supplies `masked_logprobs(logp, old_logp, ref_logp, mask)` and `token_statistics(ratio, kl, mask, clip_eps)`. The first neutralizes padding; the second returns all three detached diagnostics. Call these helpers while implementing group-relative advantages, the clipped surrogate, and per-response aggregation.

Isolate padding before exponentiation. Otherwise extreme log-probability differences at invalid positions can produce `inf*0`. Old and reference policies are fixed snapshots; stop their gradients even if callers pass differentiable tensors.

## 4. Conventions

| Item | This exercise | Common implementations |
|---|---|---|
| Standard deviation | Within-group sample standard deviation plus $10^{-4}$ | Same as group mode in [TRL v0.23.1](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py) |
| Aggregation | Token mean within each response, then response mean | DAPO averages over valid tokens |
| KL | Add k3 directly to the loss | Do not first subtract KL from rewards |
| Diagnostics | Fraction of valid-token ratios outside either bound | Some libraries count only favorable directions that actually get clipped |

The inequality $e^d-d-1\ge0$ makes the token-level KL term nonnegative. Interpreting its expectation as KL also requires the appropriate sampling distribution. T3 uses a toy policy with EOS, eight groups of eight responses, and 150 training steps to learn a reward for matching the first three characters of `abc`.

When all responses to a prompt receive the same reward, centered advantages are zero and should cause no reward-driven policy update. A KL term can still pull the policy toward the reference. Separate those two sources of gradients when debugging zero advantages.

## 5. Interactive visualizations

[Open group advantages](#viz-group_advantage): edit one reward to see how its group changes, and toggle standard-deviation normalization to inspect reward scale.

[Open the token-gradient heatmap](#viz-token_weight_heatmap): adjust response lengths and compare per-token weights in GRPO, DAPO, and Dr. GRPO. Training plots also record reward and response length.

## 6. Run your implementation

Open problem **4**, implement both functions in the browser editor, and click **Run tests**. Results appear beside the editor.

Check normalization and padding first, then snapshot gradients, and finally policy training with EOS.

## 7. Further exploration

[Dr. GRPO](../dr_grpo/README.md) changes only advantage scaling and the aggregation denominator, exposing length and difficulty bias. For the KL estimator's derivation, see [Schulman, 2020](http://joschu.net/blog/kl-approx.html).
