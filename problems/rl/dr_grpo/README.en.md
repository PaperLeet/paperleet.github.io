# Dr. GRPO Length and Difficulty Bias Corrections

## 1. What the paper solves

Dr. GRPO examines two normalizations in group-relative optimization. Averaging by response length changes per-token weights; scaling by group standard deviation changes weights across prompts. [Sections 3.1–3.2](https://arxiv.org/html/2503.20783v2#S3.SS2) describe the corresponding corrections.

Module goal: Remove group standard-deviation scaling and use a fixed length denominator to correct two normalization biases. Implement the two normalization corrections; the token objective, masking helpers, and diagnostics are supplied. Completing this exercise completes one selected core module of the paper.

## 2. Core equations and task

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `drgrpo_advantages` · Required

Return one floating Tensor `advantages`, shape `[B * G]` (`B` prompts, `G` adjacent responses each), preserving reward order, dtype and device. Values are group-centered rewards without std scaling; do not reduce or retain a separate group axis.

#### `drgrpo_loss` · Required

Return `(loss, stats)`. `loss` is a scalar Tensor, shape `()`, equal to the negative masked objective sum divided by `N * max_len` for `[N, L]` token inputs. `max_len` is the external fixed generation limit. Retain gradients to `logp`; freeze old/reference scores and advantages. The dictionary has exactly the documented fields `kl_mean`, `clipfrac`, `ratio_mean`: floating scalar Tensors of shape `()`, detached, averaged over all valid tokens. `clipfrac` counts `abs(ratio - 1) > clip_eps`. With no valid tokens, all three are zero.

#### `token_objective` · Provided

Returns `(objective, ratio, kl)`, a tuple of three floating Tensors, all `[N, L]` (responses × tokens). `objective` is the clipped advantage objective **already minus `beta * kl`**, with no mask or reduction applied. `ratio` is the importance ratio and `kl` the k3 estimate. All three retain `logp` gradient paths; old/reference scores and advantages are frozen. Apply padding isolation before calling it and aggregate the objective in `drgrpo_loss`.

#### `masked_logprobs` · Provided

Returns `(masked_logp, masked_old_logp, masked_ref_logp)` in the same order as the three score inputs. Each Tensor keeps its input's `[N, L]` shape, dtype and device, with zero at masked positions and unchanged scores elsewhere. This helper preserves active-position gradient paths; the loss caller freezes old/reference scores separately.

#### `token_statistics` · Provided

Returns a `dict[str, Tensor]` for `[N, L]` token arrays. The dictionary has exactly the documented fields `kl_mean`, `clipfrac`, `ratio_mean`: floating scalar Tensors of shape `()`, detached, averaged over all valid tokens. `clipfrac` counts `abs(ratio - 1) > clip_eps`. With no valid tokens, all three are zero.

#### `grpo_reference_loss` · Provided reference alias

The supplied GRPO baseline returns `(loss, stats)`: `loss` is a scalar Tensor `()`, averaging valid tokens within each response and then all responses. The dictionary contains detached scalar `kl_mean`, `clipfrac` and `ratio_mean`, each averaged over valid tokens. Its policy gradient remains live, with old/reference scores and advantages frozen. This baseline uses GRPO's response normalization, not the fixed `N * max_len` denominator you implement.

#### `grpo_reference_advantages` · Provided reference alias

The supplied GRPO baseline returns one floating Tensor `[B * G]`, preserving reward order, dtype and device. By default it subtracts the group mean and divides by sample standard deviation plus `eps`; your required `drgrpo_advantages` instead omits this scaling.

### Formula and conventions

Using the supplied GRPO token objective, implement `drgrpo_advantages` and `drgrpo_loss`. Rewards `[B*G]` remain contiguous by group. All three log-probability tensors and the mask have shape `[N,L]`; advantages have shape `[N]`, with $N=BG$. Group size must be at least two and divide the reward count.

$$
\hat A_i=r_i-\bar r_g,\qquad
L=-\frac{1}{N L_{max}}\sum_{i,t}m_{i,t}
\left[\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-c,1+c)\hat A_i)-\beta k3_{i,t}\right].
$$

$L_{max}$ is a fixed generation limit supplied by the caller, not the longest response in the batch. The definitions of $\rho$, $k3$, and clipping match problem **4**. Defaults are $c=0.2$ and $\beta=0.04$. Return the loss and three detached diagnostics, `kl_mean, clipfrac, ratio_mean`, averaged over valid tokens.

Consider two negatively advantaged responses of lengths five and twenty, with equal advantages. Under GRPO, their per-token penalty magnitudes differ by a factor of four: each token in the longer incorrect response receives a smaller penalty. A fixed denominator makes the per-token gradients equal. Without standard-deviation scaling, multiplying rewards by a constant also multiplies advantages by that constant, making reward scale a meaningful training parameter again.

## 3. From equations to code

```text
rewards [B*G] → [B,G] → subtract group mean → [B*G]
Call the supplied token_objective → objective/ratio/k3: [N,L]
Mask objective → sum over all tokens → divide by N*max_len
Keep valid-token normalization for diagnostics
```

The starter imports `token_objective(logp, old_logp, ref_logp, advantages, clip_eps, beta)` from problem **4**'s reference implementation. It returns `(objective, ratio, kl)`, with $-\beta k3$ already included in the objective. The starter also supplies `masked_logprobs` and `token_statistics` for padding and diagnostics. Reuse these helpers to focus on the two normalization changes; completing another exercise first is not required. Isolate padding correctly before aggregation, and do not let other responses' lengths alter the target response's denominator.

## 4. Conventions

| Item | This exercise | Paper and common implementations |
|---|---|---|
| Advantages | Subtract group mean only | Dr. GRPO also removes standard-deviation scaling |
| Fixed constant | Caller-supplied `max_len` | The paper recommends the generation budget; other constants change gradient scale |
| KL coefficient | Retain problem 4's value of 0.04 | Section 2 and Appendix G of the paper use zero |
| Empirical length trend | Display the experiment without grading one run's direction | Multiple seeds are needed before using a trend as a stable behavioral threshold |

KL is retained here to isolate the two changes for teaching. It is not the paper's training configuration. Gradient properties can be checked exactly; learned mean length also depends on EOS, learning rate, reward design, and sampling noise. Read those two kinds of evidence separately.

## 5. Interactive visualizations

[Open token weights](#viz-token_weight_heatmap): adjust two response lengths and inspect per-token weights under GRPO and Dr. GRPO.

[Open difficulty bias](#viz-difficulty_bias): vary the fraction of correct responses and compare standardized rewards with centered rewards. The [length experiment](#viz-t3_length_curves) shows incorrect-response length curves and is explicitly display-only.

## 6. Run your implementation

Open problem **5**, write both functions in the browser editor, and click **Run tests**. Review gradient checks and experiment plots in the results panel.

The length-gradient test includes problem **4**'s values for comparison. The fixed-limit test changes unrelated response lengths to rule out accidental dependence on batch composition.

## 7. Further exploration

Return to [GRPO](../grpo/README.md) to compare DAPO's valid-token average. With a fixed sampling budget, study separately how normalization affects gradient scale, reward improvement, and the lengths of incorrect responses.
