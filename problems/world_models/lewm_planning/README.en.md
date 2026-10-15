# LeWM Latent CEM Planning

In [LeWorldModel §3.2 and Appendix B](https://arxiv.org/html/2603.19312v2), a frozen world model predicts the consequences of candidate action sequences. The Cross-Entropy Method (CEM) repeatedly samples plans, keeps the lowest-cost candidates and refits a Gaussian distribution to those elites.

## Implementation requirements

### Return requirements

#### `terminal_cost` (required)

Return one floating `Tensor` `[N]`, one cost per candidate rollout `[N,H,D]`; H is the predicted horizon and D latent width. Sum squared terminal-state errors over D only, with no horizon or candidate reduction and no division by D. Preserve rollout dtype/device and input values. This sampling-based task imposes no gradient or detach requirement.

#### `cem_refit` (required)

Return `(mean, variance, elite_indices)`, exactly three Tensors in that order. For candidates `[N,H,A]` and K=`num_elites`, mean and population variance are floating `[H,A]` with candidate dtype/device; elite_indices is int64 `[K]` on the same device, ordered by ascending cost then original index on ties. Reduce over K elites only; return variance (zero for K=1), not standard deviation. No input mutation; this task imposes no gradient or detach requirement.

#### `_float_tensor` (provided)

Provided validator: return Python `None` when x is a finite, nonempty float32/float64 Tensor with rank ndim. Raise `ValueError` when a checked condition fails; do not mutate inputs or return transformed tensors.

### Formula and conventions

Implement the two kernels that decide which plans survive and how the next population is sampled.

| Function | Inputs | Outputs |
|---|---|---|
| `terminal_cost(rollouts, goal)` | Latent states `(N,H,D)`, goal `(D,)` | One cost per candidate `(N,)` |
| `cem_refit(candidates, costs, num_elites)` | Actions `(N,H,A)`, costs `(N,)`, integer `1<=K<=N` | Mean `(H,A)`, variance `(H,A)`, elite indices `(K,)` |

Use the paper's **terminal squared L2** objective:

$$J_i=\sum_d(\hat z_{i,H,d}-z_{goal,d})^2.$$

Earlier latent states do not contribute. Sum feature errors rather than averaging them. The rollout excludes the initial state and contains at least one predicted state.

Select the K smallest costs. Break cost ties by the original candidate index, using a stable ascending ordering. Refit each action coordinate independently:

$$\mu_{h,a}=\frac1K\sum_{i\in E}u_{i,h,a},\qquad v_{h,a}=\frac1K\sum_{i\in E}(u_{i,h,a}-\mu_{h,a})^2.$$

Return **variance**, not standard deviation. The supplied sampler takes its square root. This module uses the diagonal-Gaussian maximum-likelihood convention: divide by K, so one elite gives zero variance. Do not add a variance floor, momentum, clipping or cross-time covariance inside `cem_refit`.

All tensors are finite, nonempty float32/float64 with matching dtype/device. Preserve inputs. Validators are provided. No gradient contract is needed for this sampling-based planner.

For one-step candidate actions `[-1,1,3]`, costs `[4,0,1]` and `K=2`, elite indices are `[1,2]`, mean is `2` and variance is `1`.

## Toy: plan five actions toward a goal

**Teaching adaptation.** A supplied additive 2D transition `z_next=z+action` replaces the trained visual world model. A fixed seed generates 128 candidates per iteration, with 16 elites and 14 updates. Sampled actions are clamped to `[-1,1]` before evaluation; your refit sees those bounded candidates. Your functions drive the terminal-goal cost and elite updates. Plots show sample/mean-plan goal error, the final trajectory and sampling variance.

The experiment evaluates an open-loop plan. The paper's MPC wrapper executes only a prefix and replans from new observations; environment interaction and replanning are outside this module. This toy does not reproduce Push-T or the paper's runtime results. T1 checks scalar oracles, T2 checks ties and invariances, and T3 runs the actual submitted kernels.
