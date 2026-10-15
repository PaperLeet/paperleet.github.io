# Posterior and Reverse Denoising Step

## 1. What the paper solves

DDPM decomposes generation into many small denoising steps. An analytic Gaussian posterior supplies a training target; at sampling time, predicted noise determines where each step moves.

This is the second selected core module of DDPM. Implement only `q_posterior` and `p_sample_step`, connecting the analytic posterior and noise prediction into one reverse update. The full `ddpm_sample` loop and the test noise predictor are supplied.

## 2. Core equations

### Return requirements

#### `q_posterior` (required)

Return the tuple (mean, variance), both Tensors, in that order. mean has x_t shape [B,...]; variance has shape [B,1,...,1], the same rank as x_t with singleton non-batch axes, broadcastable over each sample. B is batch size. Return variance, not standard deviation or log variance. Both use x_t floating dtype/device and preserve gradients; variance is zero at index t=0.

#### `p_sample_step` (required)

Return one reverse-step sample Tensor with x_t shape [B,...] and floating dtype/device; B counts samples and trailing axes are coordinates. Return only the sample, not posterior statistics. Handle t=0 per example: neither variance option adds noise there. The function does not itself stop gradients.

#### `ddpm_sample` (provided)

With return_traj=False, return final samples as a Tensor with the requested shape=(B,...) and betas dtype/device. With True, return (samples, trajectory) in that order. trajectory is a Tensor [T+1,B,...], T=len(betas): initial Gaussian noise first, then each reverse update from index T-1 through 0, ending at samples. All returned tensors are gradient-free.

#### Provided local helpers

- `_extract`: Provided: return coefficients selected by t as a Tensor [B,1,...,1], with x rank and singleton non-batch axes, B=len(t). Convert to x dtype/device while preserving differentiable paths through values. Leave all inputs unchanged.

### Equations and interface notes

Given the clean data, $q(x_{t-1}\mid x_t,x_0)$ is Gaussian. Its mean is a weighted combination of the current sample and clean data:

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

The network cannot observe the true x0, so first reconstruct it from the noise prediction:

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad x_{t-1}=\tilde\mu_t(x_t,\hat x_0)+\sigma_tz.$$

$x_t$, $\hat\epsilon$, and $z$ have shape `[B,...]`; t has shape `[B]`, allowing a different time for each row. The variance has shape `[B,1,...]` and broadcasts to the sample. See [DDPM Equations 6, 7, and 11, and Algorithm 2](https://arxiv.org/abs/2006.11239).

This exercise replaces the network with the exact score of an eight-component Gaussian mixture, giving $\epsilon^*=-\sqrt{1-\bar\alpha_t}\nabla\log p_t$. Sampling quality can then be checked without training. Finite-step discretization and finite-sample error remain, so empirical MMD need not equal zero.

## 3. From equations to code

Implement two core functions:

```text
q_posterior: extract coefficients at [B] times; combine x0 and x_t
Return the posterior mean and variance
p_sample_step: recover x0 from epsilon, then call q_posterior
Choose posterior or beta variance according to var_type
Zero the noise contribution wherever t=0 and return the updated sample
```

`q_posterior` returns a mean of shape `[B,...]` and broadcastable variance `[B,1,...]`. `p_sample_step` returns the shape and dtype of `x_t`. Raise `ValueError` for an unknown `var_type`.

`_extract` and `ddpm_sample` are supplied. The sampler handles initial Gaussian noise, descending times, the caller’s generator, and trajectory collection; each update calls your `p_sample_step`. With `return_traj=True`, it returns `(samples, trajectory)` with shape `[T+1,B,...]` including initial noise.

## 4. Conventions

| Item | This exercise | improved-diffusion |
| --- | --- | --- |
| x0 range | No clipping for two-dimensional data | Images are often clipped to [-1,1] |
| Variance options | posterior or beta | Also supports learned variance |
| Final step | No added noise under either option | Also masks the final noise |
| First log variance | No log variance is computed | Replaces the first variance with the next value before taking logs |

## 5. Interactive visualizations

<viz id="reverse_particles" />

Play the reverse trajectory and switch between posterior and beta variance. Observe particles move from noise into eight modes; gray centers mark the true data locations.

<viz id="variance_compare" />

Move through time to compare beta and posterior variance. Their final-step difference is greatest, but both options must stop adding noise there. After running your implementation, the results panel can overlay generated points, true samples, and the MMD threshold.

## 6. Run your implementation

Open module **8** in the browser, complete `q_posterior` and `p_sample_step`, and run the tests.

T2 derives the posterior independently through Gaussian precision addition. T3 connects your update to the supplied loop, checking both variance choices with 1,000 steps and 2,000 samples. The MMD threshold is 1.5 times the fixed-seed reference result; mode proportions and outliers are also checked.

## 7. Further exploration

DDIM reuses the same noise predictor while skipping along a subsequence of times. For learned variance, continue with [Improved DDPM](https://arxiv.org/abs/2102.09672).
