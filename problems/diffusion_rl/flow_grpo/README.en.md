# Flow-GRPO: stochastic flow transitions and group-relative updates

[Flow-GRPO](https://arxiv.org/abs/2505.05470) turns flow-matching generation into a stochastic policy that can be optimized with group rewards. Implement the reverse-time Gaussian transition and the corresponding clipped, KL-regularized update.

## 1. Convert the flow ODE into a Gaussian policy

Use the paper's data-to-noise time convention $x_t=(1-t)x_0+t\epsilon$ and velocity pointing toward noise. Sampling moves to a smaller time: $\Delta t=s-t<0$. For noise level $a\geq0$, define

$$
\sigma_t^2=a^2\frac{t}{1-t},\qquad
b_\theta=v_\theta+\frac{\sigma_t^2}{2t}\left(x_t+(1-t)v_\theta\right).
$$

The reverse-time Euler–Maruyama transition is

$$
\mu_\theta=x_t+\Delta t\,b_\theta,
\qquad q=\sigma_t\sqrt{-\Delta t},
\qquad x_s=\mu_\theta+qz,\quad z\sim\mathcal N(0,I).
$$

Implement `sde_transition(state, velocity, time, next_time, noise_level)` and return `(mean, std)`. `state` and `velocity` have the same nonempty floating shape `(B, ...)`. The output mean preserves that shape, while `std` is a scalar tensor of the same dtype/device. Require `0 <= next_time < time < 1`; endpoints with singular score/noise formulas are deliberately excluded. A zero noise level returns the ODE mean and zero standard deviation.

Detach the observed state; gradients flow through the supplied velocity. This is a local policy update conditioned on fixed rollout states. The square root uses the magnitude of the reverse-time step, matching the [authors' sampler](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py).

## 2. Group advantages and per-transition ratios

Implement `flow_grpo_loss(current, old, reference, actions, std, rewards, clip_epsilon, beta, eps=1e-8)` with these shapes:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `current`, `old`, `reference` | `(P,G,T,D)` | Current, rollout and reference transition means |
| `actions` | `(P,G,T,D)` | Previously sampled next states |
| `std` | `(T,)` | Shared positive Gaussian standard deviation at each step |
| `rewards` | `(P,G)` | Terminal rewards for G trajectories of each prompt |

There must be at least two trajectories per prompt. Normalize rewards **within each prompt** using population standard deviation:

$$
A_{pg}=\frac{R_{pg}-\overline R_p}
{\sqrt{\frac1G\sum_j(R_{pj}-\overline R_p)^2}+\varepsilon_{\rm std}}.
$$

Repeat this terminal advantage over the trajectory's transitions. Constant-reward groups yield zero advantage. The exact equal-variance Gaussian log-density ratio is

$$
\log r_{pgt}=\sum_d\frac{(a_{pgtd}-\mu_{\rm old,pgtd})^2-(a_{pgtd}-\mu_{\theta,pgtd})^2}{2q_t^2}.
$$

The normalization constants cancel because the old and current distributions share the variance. Compute the exact KL to the fixed reference:

$$
K_{pgt}=\sum_d\frac{(\mu_{\theta,pgtd}-\mu_{\rm ref,pgtd})^2}{2q_t^2}.
$$

Return the scalar loss

$$
L=\frac1{PGT}\sum_{p,g,t}\left[
-\min\left(r_{pgt}A_{pg},\operatorname{clip}(r_{pgt},1-\epsilon,1+\epsilon)A_{pg}\right)
+\beta K_{pgt}\right].
$$

Only `current` receives gradients. Detach old/reference means, sampled actions, rewards and standard deviations inside the loss.

## 3. Exact conventions

This exercise implements the **joint Gaussian density** from the paper's policy ratio: sum over latent coordinates, then average the objective over prompts, trajectories and time. The official sampler averages coordinate log densities as an implementation normalization; that would produce a geometric-mean ratio and is intentionally not the convention here. The exact KL is also a coordinate sum.

Use finite `0 < clip_epsilon < 1`, `beta >= 0`, `eps > 0` and strictly positive `std`. Reject invalid shapes, non-floating or empty tensors, mismatched dtype/device and non-finite inputs with `ValueError`. Preserve CPU float32/float64 and never mutate inputs. Handle zero advantages and already-clipped positive ratios without overflow; remaining nonzero objective terms fit the floating range. A zero-noise ODE transition has no Gaussian density and cannot be passed into this loss.

## 4. Verification

T0 checks interfaces and invalid schedules. T1 uses independent scalar loops for transition statistics, Gaussian densities, group normalization, clipping and KL. T2 checks the score-corrected velocity Jacobian, analytic policy gradients, frozen rollout context, independent prompt reward offsets, temporal averaging and both signs of clipped advantage. T3 runs 80 actual online updates with fresh stochastic trajectories from the current policy.

## 5. Online experiment

Three scalar prompts define target next states. A single trainable weight controls a conditional velocity, each prompt samples 24 transitions, and terminal rewards are negative squared target errors. The experiment uses your SDE statistics to sample and your loss to update the policy. The chart tracks the target error of the expected next state. This CPU teaching adaptation focuses on one transition per trajectory; full trajectory handling is covered numerically in T1/T2. It does not reproduce SD3 image training or the paper's denoising-reduction benchmark.

## 6. Run

```bash
paperleet init diffusion_rl/flow_grpo
paperleet test diffusion_rl/flow_grpo
```

## 7. Sources

- Liu et al., [Section 4.1, Eqs. (4)–(6); Section 4.2, Eqs. (9)–(13)](https://arxiv.org/html/2505.05470v1).
- [Official reverse-time sampler](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py).

The compact implementation is written independently from the mathematical specification.
