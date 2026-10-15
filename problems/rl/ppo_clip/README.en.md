# PPO Clipped Policy Objective

## 1. What the paper solves

Training for multiple epochs on the same trajectories can move the policy far from the distribution that collected them. PPO uses a pessimistic clipped surrogate to support simple first-order updates. See [Equations 7 and 9](https://arxiv.org/pdf/1707.06347).

Module goal: Implement the pessimistic clipped objective and verify gradients for both advantage signs. Implement only clipped_surrogate; value loss, entropy, normalization, and training are supplied. Completing this exercise completes one selected core module of the paper.

## 2. Core equations and task

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `clipped_surrogate` · Required

Return one floating scalar Tensor `loss`, shape `()`: the **negative batch mean** of the clipped policy surrogate for `[B]` inputs. Preserve input floating dtype/device. Retain gradients through `new_logp`; freeze `old_logp` and `advantages`. Return the Tensor itself, without `.item()` or `.detach()`.

#### `ppo_loss` · Provided

Returns a `dict[str, Tensor]` with all six keys below; every value has scalar shape `()`:

- `loss`: total minimized loss, `pg_loss + vf_coef * v_loss - ent_coef * entropy`.
- `pg_loss`: clipped policy loss; `v_loss`: half the mean value error (with optional value clipping); `entropy`: batch mean, or zero if omitted.
- `approx_kl`: detached mean k3 KL estimate; `clipfrac`: detached fraction with `abs(ratio - 1) > clip_eps`.

The loss terms retain their current-policy/value gradients; old policy, old values, advantages and returns are fixed targets. Read the objective as `result["loss"]`, not by tuple position.

### Formula and conventions

Implement only `clipped_surrogate(new_logp, old_logp, advantages, clip_eps=0.2)`. All three inputs have shape `[B]`; return one scalar policy loss to minimize. Advantages are already prepared. Detach `old_logp` and `advantages` inside this function while retaining gradients through `new_logp`.

$$
r=e^{\log\pi_\theta-\log\pi_{old}},\qquad
L^{PG}=-\mathbb E[\min(r\hat A,\operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)].
$$

For positive advantages, a ratio above the upper bound earns no further improvement. For negative advantages, a ratio below the lower bound earns no further improvement. Movement in the unfavorable direction still incurs a penalty. Take the smaller of the unclipped and clipped surrogate values.

The starter supplies a complete `ppo_loss` adapter. It calls your function and handles advantage normalization, value clipping, entropy, and diagnostics for the existing bandit experiment. Keep that supplied code in place.

## 3. From equations to code

```text
Three [B] vectors → detach old_logp and advantages
Exponentiate the log-probability difference → ratio [B]
Compute ratio * advantage and clipped_ratio * advantage
Take the elementwise minimum → negative batch mean → scalar
```

Check all four combinations of advantage sign and ratios beyond the clipping interval. The supplied adapter consumes your scalar policy loss, so completing this function is sufficient for the full test suite.

## 4. Supplied training helpers and conventions

| Item | Supplied behavior |
|---|---|
| Advantage normalization | Sample standard deviation plus $10^{-8}$; zero standard deviation for one sample |
| Value loss | Optional value clipping; half the maximum of the two squared errors |
| Entropy | Optional entropy bonus; zero when absent |
| Diagnostics | Detached `approx_kl = mean(r-1-log(r))` and `clipfrac` |

Integration tests retain these established conventions. Your implementation work is the clipped policy objective. PPO clipping does not impose a strict KL bound; the bandit threshold applies to this fixed learning rate and batch.

## 5. Interactive visualizations

[Open the clipped objective](#viz-clip_objective): adjust $\epsilon$, compare positive and negative advantages, and move the probability ratio to inspect gradient directions and flat regions.

[Open the trust-region experiment](#viz-trust_region): compare per-epoch KL for the reference, a ratio-clipping-only implementation, and an incorrect `max` implementation.

## 6. Run your implementation

Open this paper's **PPO Clipped Policy Objective** module, fill in `clipped_surrogate`, keep the supplied `ppo_loss`, and run the tests. Inspect the core-function and four-quadrant gradient checks before the supplied bandit experiment.

For local editing, run `paperleet init rl/ppo_clip`, edit `workspace/rl/ppo_clip.py`, and run `paperleet test rl/ppo_clip`.

## 7. Further exploration

[DPO](../dpo/README.md) derives a policy objective directly from preferences. You can also study TRPO to compare constrained optimization with PPO's surrogate objective.
