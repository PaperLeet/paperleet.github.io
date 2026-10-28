# SuperMimic: Differentiable Imagination Returns and Critic Targets

## 1. Exercise goal

Implement the return calculations in [SuperMimic](https://arxiv.org/abs/2610.28480), Section 3.3: a differentiable actor objective and frozen critic targets. This is the 2026 world-model method, with [the authors' project page](https://supermimic.github.io/).

## 2. Interfaces and returns

| Function | Inputs | Return |
|---|---|---|
| `imagined_return(rewards, termination, bootstrap, gamma)` | `(B,H)` rewards and probabilities, `(B,)` terminal values, scalar discount | Scalar Tensor: batch-mean objective to **maximize** |
| `lambda_return_targets(rewards, termination, values, gamma, lam)` | `(B,H)` rewards/probabilities, `(B,H+1)` critic values, discount and lambda | Frozen `(B,H)` Tensor of critic targets |

All tensors are finite CPU float32/float64, share dtype/device, and have positive batch and horizon sizes. Preserve inputs and dtype/device. `gamma` is finite in `[0,1)`; `lam` is finite in `[0,1]`; termination probabilities are in `[0,1]`. `_rollout` is a provided validation helper returning `None` or raising `ValueError`.

### Return requirements and indexing

Column `t` of `rewards` is the **arrival reward** `r_(t+1)`, and column `t` of `termination` is `d_(t+1)`. Column `t` of `values` is `V_t`, so the last column is `V_H`.

The actor output retains gradient paths through rewards, termination probabilities and terminal values. Freezing world-model and critic **parameters** is the caller's job; their state/action derivatives must remain available to the policy. Do not detach the terminal value just because the critic weights are fixed. The target function detaches **all three** tensor inputs internally and returns `requires_grad=False` targets.

## 3. Equations

Start with survival `c_0=1` and update `c_(t+1)=c_t*(1-d_(t+1))`. The actor maximizes:

$$
\hat J=\frac1B\sum_b\left[\sum_{t=0}^{H-1}\gamma^t c_{b,t}r_{b,t+1}+\gamma^H c_{b,H}V_{b,H}\right].
$$

Count the arrival reward before applying its termination probability to future rewards. Keep survival probabilities soft and differentiable.

For critic targets, initialize `R_H=V_H`, then recurse backward:

$$
R_t=r_{t+1}+\gamma(1-d_{t+1})\left[(1-\lambda)V_{t+1}+\lambda R_{t+1}\right].
$$

Return targets for `t=0,...,H-1` in chronological order, without the initialization `R_H` column. Detach these targets for subsequent critic fitting. At `lambda=0`, this is one-step bootstrapping; at `lambda=1`, the first target agrees with the actor's complete bootstrapped return for that example.

## 4. Worked example

With rewards `[2,3,100]`, termination `[0,1,0]`, `gamma=.5` and any terminal value, the return is `2+.5*3=3.5`. The terminating transition still earns `3`; the later reward and terminal value are excluded. Soft probabilities preserve partial future contributions.

## 5. Deterministic experiment

Four scalar initial states share a feedback policy `action=gain*(goal-state)`. Fixed differentiable dynamics advance each state for eight steps; analytic heads provide tracking rewards, termination probabilities and terminal values. Gradient ascent through your actor objective learns the gain. Your lambda targets are independently checked against the full-return identity at `lambda=1`.

This is a teaching adaptation of the return mechanism. It uses a fixed analytic world model, rather than fitting the paper's network, and does not reproduce humanoid tracking, replay collection or sim-to-real results.

## 6. Run your implementation

Complete the two functions in `starter.py`, then run through PaperLeet. T0 checks input/return contracts, T1 uses scalar recursions, T2 tests finite-difference gradients and horizon boundaries, and T3 trains the feedback policy and records charts.

## 7. Sources

- [SuperMimic paper](https://arxiv.org/abs/2610.28480): Section 3.3, Eq. (2) actor objective and Eq. (3) critic targets.
- [Official project](https://supermimic.github.io/): Qiayuan Liao et al., CoRL 2026.
