# DPO Preference Objective and Implicit Rewards

## 1. What the paper solves

DPO rewrites KL-regularized reward maximization as a classification objective on preference pairs, eliminating the need to fit a separate reward model during training. [Section 4](https://arxiv.org/html/2305.18290v3#S4) derives the equivalence from the optimal policy.

Module goal: Compute the preference loss and implicit rewards from policy and reference sequence scores. Implement only dpo_loss; sequence log-prob aggregation and the toy training experiment are supplied. Completing this exercise completes one selected core module of the paper.

## 2. Core equations and task

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `dpo_loss` · Required

Return `(loss, chosen_rewards, rejected_rewards)`, a tuple of **three Tensors in this order**:

- `loss`: shape `()`, the batch-mean smoothed DPO loss; retain policy gradients.
- `chosen_rewards`: shape `[B]`, `beta * (pi_chosen - ref_chosen)`, detached when returned.
- `rejected_rewards`: shape `[B]`, `beta * (pi_rejected - ref_rejected)`, detached when returned.

`B` is the number of preference pairs. Preserve input floating dtype/device and keep reward vectors shaped `[1]` for `B=1`. Reference inputs are fixed. Do not detach the policy computation used for the loss, call `.item()` on the loss, or average/squeeze the returned reward vectors. Returning only `loss` does not satisfy the interface.

#### `sequence_logprob` · Provided

Returns one floating Tensor `scores`, shape `[B]`, from logits `[B, T, V]` (`T` positions, `V` vocabulary entries). Each value is the **sum** of shifted, masked response-token log-probabilities, not a token mean. Preserve logits dtype/device and gradients; `B=1` still returns `[1]`.

### Formula and conventions

Implement only `dpo_loss`. The supplied `sequence_logprob` helper handles next-token label shifting, response masking, and summation to produce the sequence scores.

The four sequence log-probability inputs have shape `[B]`, where `B` is the number of preference pairs.


$$
h=(\log\pi_c-\log\pi^{ref}_c)-(\log\pi_r-\log\pi^{ref}_r),\qquad
L=-\mathbb E[(1-\varepsilon)\log\sigma(\beta h)+\varepsilon\log\sigma(-\beta h)].
$$

Here $c$ and $r$ mean chosen and rejected. Defaults are $\beta=0.1$ and label smoothing $\varepsilon=0$, giving the original paper objective. Nonzero smoothing is an optional compatibility parameter. The implicit reward is $\beta(\log\pi-\log\pi^{ref})$. It measures improvement relative to the reference policy, so the chosen response's raw log-probability alone is not a predicted preference score.

## 3. From equations to code

```text
Supplied sequence_logprob → four [B] sequence-score vectors
Detach the reference; compute chosen/rejected rewards and their margin
Combine positive and reversed labels with stable log-sigmoid → scalar loss
```

Keep the computation in log space. When $\beta h=-10000$, computing sigmoid first underflows to zero, and taking its logarithm produces infinity. A stable function evaluates the same objective directly, without clipping the margin or erasing gradients for hard examples.

## 4. Conventions

| Item | This exercise | Related methods or libraries |
|---|---|---|
| Token aggregation | Sum of response log-probabilities | SimPO uses length normalization, giving a different objective |
| Mask | Aligned with shifted labels | Excludes both prompt and padding tokens |
| Label smoothing | Convex combination of forward and reversed sigmoid losses | Conservative DPO as described in TRL v0.23.1 |
| Reference and returned rewards | No reference gradients; detached returned rewards | Rewards are diagnostic outputs, not another backward path |

[TRL v0.23.1 documentation](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) connects this smoothing to conservative DPO. It differs from the label-noise correction used by the `robust` loss. T3 uses six candidates, a fixed nonuniform reference, and 20,000 Bradley–Terry preference pairs. It compares the trained policy with $\pi^*\propto\pi_{ref}e^{r^*/\beta}$ and checks that implicit rewards exactly recover the true ranking.

If the reference model already favors one response, the policy must improve its relative probability by more to create a positive preference margin. This explains both the reference distribution's role in the closed-form optimum and the experiment's use of a nonuniform reference.

## 5. Interactive visualizations

[Open the label and mask grid](#viz-shift_mask_grid): inspect which next token each position predicts and which contributions the prompt mask excludes.

[Open the margin curves](#viz-loss_vs_margin): adjust $\beta$ and $\varepsilon$ to inspect the loss and gradient weights. Reward scatter plots show how training recovers the true reward ranking.

## 6. Run your implementation

Open this paper's **DPO Preference Objective and Implicit Rewards** module, implement `dpo_loss`, and keep the supplied `sequence_logprob`. Run the tests and inspect the reference-adjusted margin, beta scaling, gradient, and extreme-value checks.

Sequence tests check the supplied helper's integration. The behavioral test supplies training for a small categorical policy.

## 7. Further exploration

Continue with [GRPO](../grpo/README.md) to compare offline preference learning with online group sampling. You can also investigate which optimality assumptions change under length normalization or noisy preferences.
