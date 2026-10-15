# DDIM Subsequence Update

## 1. What the paper solves

DDPM's training objective specifies noisy marginal distributions at each time without uniquely determining sampling trajectories. DDIM uses this freedom to reuse an existing noise predictor and generate samples in fewer denoising steps.

Implement only `ddim_step`, the paper’s central update across a timestep subsequence. Timestep selection and the full sampling loop are supplied. Inversion helpers and visualizations remain available for optional exploration.

## 2. Core equations

### Return requirements

#### `ddim_step` (required)

Return the sample Tensor at the less noisy destination, matching x_t shape [B,...] and floating dtype/device; B is batch size and trailing axes are sample dimensions. Do not return the clean prediction or coefficients. eta=0 removes dependence on noise; ab_prev=1 returns the clean endpoint. This step does not itself stop gradients.

#### `ddim_timesteps` (provided)

Return a torch.long Tensor of shape [num_steps] (CPU by default here), containing strictly descending discrete indices starting at T-1. Each entry corresponds to one model call. The final clean boundary is not an extra entry, and index 0 need not occur. The output has no gradients.

#### `ddim_sample` (provided)

With return_traj=False, return final samples as a Tensor [B,...], matching x_T shape and dtype/device. With True, return the tuple (samples, trajectory) in that order. trajectory is a Tensor [num_steps+1,B,...]: entry 0 copies x_T, each following entry is one update, and the last equals samples. Both outputs are gradient-free; x_T is unchanged.

#### `ddim_invert` (provided)

Return the inverted noisy endpoint Tensor with x0 shape [B,...] and dtype/device, at the selected schedule maximum index T-1. Return only the endpoint, not a trajectory. Leave x0 unchanged; the output is gradient-free. Discrete inversion is approximate, not a promise of bitwise sampling reversibility.

### Equations and interface notes

Let the current cumulative signal be $a=\bar\alpha_t$ and the previous sampling time's cumulative signal be $b=\bar\alpha_{\mathrm{prev}}$, with $b\ge a$. Reconstruct the clean prediction, then combine a deterministic direction and random noise:

$$\hat x_0=\frac{x_t-\sqrt{1-a}\hat\epsilon}{\sqrt a},\qquad\sigma=\eta\sqrt{\frac{1-b}{1-a}}\sqrt{1-\frac ab}.$$

$$x_{\mathrm{prev}}=\sqrt b\hat x_0+\sqrt{1-b-\sigma^2}\hat\epsilon+\sigma z.$$

Sample tensors have shape `[B,D]`; a and b may be scalars or broadcastable to the sample shape. At eta=0, initial noise uniquely determines the trajectory. At eta=1 with adjacent timesteps, the stochastic variance equals the DDPM posterior variance. Subtract the stochastic variance from the direction term to avoid adding too much total noise. See [DDIM Equations 12 and 16](https://arxiv.org/abs/2010.02502) for the update and eta parameterization.

Set b=1 on the final update to reach the true clean endpoint. A point distribution provides a strong check: with perfect x0 predictions, both one step and fifty steps should arrive exactly at the point.

## 3. From equations to code

Implement only `ddim_step(x_t, eps_pred, ab_t, ab_prev, eta, noise)`:

```text
Recover x0 from the current ab_t and epsilon prediction
Compute sigma from the current and target cumulative coefficients
Subtract sigma² from the direction variance
Combine the clean prediction, epsilon direction, and supplied noise
```

Return the shape and dtype of `x_t`. `ab_t` and `ab_prev` broadcast to the sample. The final clean endpoint uses `ab_prev=1`, eliminating both the direction and random terms. With `eta=0`, the result is independent of the supplied noise.

`ddim_timesteps` and `ddim_sample` are provided. The sampler calls your `ddim_step` for each pair of times and can return a `[n+1,B,D]` trajectory including both endpoints. `ddim_invert` is also supplied for optional round-trip experiments; you do not need to implement it.

## 4. Conventions

| Item | This exercise | diffusers DDIMScheduler |
| --- | --- | --- |
| Trailing step spacing | Integer spacing T//n | Uses real-valued T/n before rounding |
| When T is divisible by n | The two sequences agree | The two sequences agree |
| Other options | Fixed trailing spacing only | Also supports leading and linspace |
| Clean boundary | Final b=1 | Controlled by set_alpha_to_one |

The original exercise checked this comparison against the [official scheduler source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) on 2026-10-13. Follow this exercise's definition when T is not divisible by n.

## 5. Interactive visualizations

<viz id="eta_steps_grid" />

Choose an eta and step-count combination. Play trajectories from the same initial noise and compare final MMD to inspect the relationship between randomness and computation.

<viz id="inversion_demo" />

Choose a data point and step count to inspect inverted noise, reconstructed positions, and round-trip error. A finer time grid usually brings the reconstruction closer to the original sample.

## 6. Run your implementation

Open module **9** in the browser, complete only `ddim_step`, and run the tests.

Core checks cover determinism, the point-distribution clean endpoint, and posterior variance. Integration checks use the supplied helpers for timestep selection, inversion, and convergence. T3 checks fifty-step generation quality with an eight-Gaussian oracle and records eta=1 as a comparison.

## 7. Further exploration

The supplied inversion is a first-order approximation. Its first call from the clean endpoint uses index zero; later calls evaluate epsilon at the current state and time. At finite step counts, it is generally not the exact algebraic inverse of a nonlinear sampling map. Compare round-trip errors in the visualization without implementing additional code. For few-step error, see [Progressive Distillation](https://arxiv.org/abs/2202.00512).
