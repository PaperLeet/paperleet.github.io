# GAE Backward Recurrence

## 1. What the paper solves

Policy gradients need to estimate how much better an action is than the current policy's average choice. GAE connects one-step temporal-difference estimates with Monte Carlo returns, using one parameter to control the estimator's bias and variance. See [Section 3, Equations 16–18](https://arxiv.org/html/1506.02438v6#S3).

Module goal: Implement the backward GAE recurrence from TD residuals with correct episode boundaries. Implement advantages and value targets; trajectories and value estimates are supplied. Completing this exercise completes one selected core module of the paper.

## 2. Core equations and task

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `compute_gae` · Required

Return `(advantages, returns)`, a tuple of two floating Tensors in that order. Both have shape `[T, N]`: `T` time steps and `N` environments, preserving `rewards` dtype/device. `returns = advantages + values`; keep both axes even when `T = N = 1`. Return the arrays, with no batch/time reduction.

### Formula and conventions

Implement `compute_gae`, returning advantages and value targets. Rewards $r_t$, current values $V_t$, and next-state values $V_{t+1}$ have shape `[T,N]`, with time first. Both episode-ending flags are boolean tensors of shape `[T,N]`.

$$
\delta_t=r_t+\gamma(1-\mathrm{term}_t)V_{t+1}-V_t,
\qquad
\hat A_t=\delta_t+\gamma\lambda(1-\mathrm{done}_t)\hat A_{t+1}.
$$

Here $\mathrm{done}=\mathrm{term}\lor\mathrm{trunc}$, the initial carry is $\hat A_T=0$, and the returned pair is $(\hat A,\hat A+V)$. The default values are $\gamma=0.99$ and $\lambda=0.95$. Gamma discounts future rewards and changes the objective's time horizon; lambda weights distant TD residuals and adjusts the estimator. They serve different purposes. At $\lambda=0$, only the current residual remains. At $\lambda=1$, intermediate value terms cancel, leaving the full return with a final-state bootstrap.

## 3. From equations to code

```text
Inputs r, V, next_V, term, trunc: [T,N]
Compute delta: [T,N]
Initialize carry: [N], all zeros
Iterate backward over t = T-1, ..., 0:
    Current residual + discounted next-step carry → A[t]
Return A and A+V: two tensors of shape [T,N]
```

Keep environments separate. Each column has its own episode boundaries; a vector operation can update all environments together, but the time axis must follow the backward dependency. Match the rewards' dtype and device, including when creating the initial carry.

## 4. Conventions

| Item | This exercise | Common alternatives |
|---|---|---|
| Termination | No future value; do not bootstrap | Usually the same |
| Time truncation | Keep the true final-state value, but stop the recursion | Some rollout examples merge the flags and compensate separately |
| `next_values` | Supplied explicitly for every step | May instead combine `values[t+1]` with a final value |
| Returns | Always $\hat A+V$ | Not a pure Monte Carlo return for arbitrary $\lambda$ |

T3 uses dynamic programming to compute exact $Q$ and $V$ for a seven-state random walk. It then perturbs the value function and compares the first action's policy-gradient estimates on the same 4,000 trajectories. Bias is measured against the exact gradient. The plotted variance is the variance of a single-trajectory estimator; the mean advantage alone cannot establish that a policy-gradient estimator is unbiased.

## 5. Interactive visualizations

[Open the GAE timeline](#viz-gae_timeline): adjust $\gamma$ and $\lambda$ to see how residuals $\delta_t$ propagate backward into $\hat A_t$, including both kinds of episode boundary.

[Open the bias–variance curves](#viz-bias_variance_vs_lambda): move $\lambda$ and compare bias, variance, and the Monte Carlo endpoint on the same trajectories.

## 6. Run your implementation

Open problem **1** in the problem list, write `compute_gae` in the browser editor, and click **Run tests**. Results and diagnostic plots appear alongside your code.

Start with $\lambda=0$ and single-step boundaries, then check episode isolation and the behavioral experiment.

## 7. Further exploration

[PPO](../ppo_clip/README.md) uses advantages in a policy objective. Explore how value-function approximation errors affect optimization, and how bootstrapping a truncated rollout differs from handling a true terminal state.
