# PKPO Reward Transform and Unbiased Policy Gradient

Implement the reward transform that optimizes the best result among `k` policy samples, following [PKPO v5, Eq. (12) and Eq. (33)](https://arxiv.org/html/2505.15201v5). Rewards can be real numbers; for binary rewards the expected maximum is pass@k.

## Functions

### Return requirements

Use the return structure below exactly; Tensor shapes include singleton dimensions. A scalar Tensor has shape `()`, not a Python `float`.

#### `maxk_estimate` · Required

Return one floating Tensor `estimate` with shape `rewards.shape[:-1]`: remove only the sample axis `N` from rewards `[..., N]`. Each value is the expected maximum over uniformly chosen size-`k` subsets. `[N]` input returns a scalar Tensor `()`, while `[B, N]` returns `[B]`. Preserve reward dtype/device and do not mutate inputs.

#### `pkpo_weights` · Required

Return one detached floating Tensor `weights`, with exactly the same `[..., N]` shape, dtype, device and original sample order as `rewards`. These are Eq. 33 coefficients, already including sample normalization; `k=1` returns `rewards / N`. Keep the sample axis even when `N=1`.

#### `pkpo_loss` · Required

Return one scalar floating Tensor `loss`, shape `()`: sum the negative weighted log-probabilities along the sample axis `N`, then average the remaining leading groups. Weights already include sample normalization; do not divide by `N` again. Keep gradients to `log_probs`, with no gradients to rewards/weights.

#### `_check` · Provided

Returns `None` after validating floating rewards `[..., N]`, `N>0`, and integer `1 <= k <= N`. Invalid inputs raise `ValueError`; it does not transform or return rewards.

#### `_comb` · Provided

Returns a Python `int`: the binomial coefficient for `0 <= k <= n`, otherwise `0`. This supplied arithmetic helper does not return a Tensor.

### Formula and conventions

Inputs are floating tensors `[..., N]`, one independent group on the last axis, `1 <= k <= N`.

1. `maxk_estimate(rewards, k)` returns `[...]`. It averages `max(rewards[I])` over all size-k subsets. For ascending sorted rewards `g[0:N]`, it equals

$$\widehat M_k=\sum_{j=0}^{N-1}\frac{\binom{j}{k-1}}{\binom Nk}g_j.$$

2. `pkpo_weights(rewards, k)` returns **detached** coefficients in original sample order:

$$a_i=\frac{1}{\binom Nk}\sum_{I:\ |I|=k,\ i\in I}\left[\max_{j\in I}g_j-\max_{j\in I\setminus\{i\}}g_j\right].$$

For `k=1`, explicitly use `a_i=g_i/N` (zero baseline for the empty set). For `k>=2`, a sorted-prefix implementation avoids subset enumeration:

$$a_{(i)}=\sum_{j<i}\frac{\binom{j}{k-2}}{\binom Nk}(g_{(i)}-g_{(j)}).$$

Set invalid binomial coefficients to zero. Ties need no arbitrary advantage: the tied differences vanish. Sort, compute prefix sums, then restore original sample order.

3. `pkpo_loss(log_probs, rewards, k)` returns `-mean_groups(sum_i(log_probs_i * a_i))`. `log_probs` has the same shape and represents the sampled actions' scalar log probabilities. Coefficients **already include sample normalization**; do not take another mean over N. Gradients flow through log probabilities only.

Supplied validation and binomial helpers can be reused. Inputs are finite CPU float32/float64 with nonempty leading groups. Preserve dtype, device, values and strided inputs. The scalar objective may differentiate through rewards; the policy weights must not. Support negative rewards, ties, `N=1`, `k=N`, and `N=256` without integer overflow.

Example: `rewards=[.2,.9,.4], k=3` gives weights `[0,.5,0]`. These are policy-gradient coefficients, not probabilities or centered GRPO advantages.

## Paper-native toy

Recreate §5.1: `X ~ Normal(theta, std=.1)`, reward `X²` inside `[0,1]`, zero outside. T3 computes expected-maximum curves and actual gradients of your policy loss for `k=1,4,8`. This independently recomputes the paper's toy setup on a smaller fixed CPU grid (2048 groups, N=16, 15 means); it does not reproduce exact published samples or train an LLM.
