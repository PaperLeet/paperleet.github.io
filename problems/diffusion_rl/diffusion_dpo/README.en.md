# Diffusion-DPO: paired denoising preference optimization

Implement the training objective from [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908), using the feature-mean error and half-beta scale of the [authors' implementation](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py). The exercise isolates the pairwise training kernel and a tiny trainable denoiser.

## 1. Paired forward noising

Each row contains a preferred clean latent $x^w$, a rejected latent $x^l$, one shared noise sample $\epsilon$, and one cumulative schedule value $\bar\alpha$. Both sides use the same timestep and noise:

$$
x_t^w=\sqrt{\bar\alpha}\,x^w+\sqrt{1-\bar\alpha}\,\epsilon,
\qquad
x_t^l=\sqrt{\bar\alpha}\,x^l+\sqrt{1-\bar\alpha}\,\epsilon.
$$

Implement `paired_noising(winner, loser, noise, alpha_bar)`. The first three tensors share a nonempty floating shape `(B, ...)`, with at least one feature axis. `alpha_bar` has shape `(B,)` and values in `[0, 1]`. Broadcast over every feature axis and return a tuple `(noisy_winner, noisy_loser)`. These are frozen training inputs: detach all four arguments before computing them. The endpoints must return pure noise at zero and the clean latent at one.

## 2. Reference-relative preference objective

Given current predictions $\epsilon_\theta^w,\epsilon_\theta^l$ and frozen reference predictions $\epsilon_{\rm ref}^w,\epsilon_{\rm ref}^l$, define feature-mean errors:

$$
e_\theta^w=\frac1D\sum_{d=1}^D(\epsilon_{\theta,d}^w-\epsilon_d)^2,
\qquad
\Delta_\theta=e_\theta^w-e_\theta^l,
\qquad
\Delta_{\rm ref}=e_{\rm ref}^w-e_{\rm ref}^l.
$$

The margin and loss are

$$
m=-\frac\beta2(\Delta_\theta-\Delta_{\rm ref}),
\qquad
L=\frac1B\sum_{i=1}^B\operatorname{softplus}(-m_i).
$$

Implement `diffusion_dpo_loss(model_w, model_l, ref_w, ref_l, noise, beta)`. Return one scalar tensor. All five tensors have the same shape, dtype and device. Use every non-batch axis for the error mean. Only `model_w` and `model_l` receive gradients; detach references and noise inside the function even when callers mark them trainable. Use a numerically stable softplus or log-sigmoid expression.

Here `beta` is a finite positive effective scale. The paper's timestep/weighting constants are absorbed into that scale; the explicit `1/2` and feature mean follow the released training code. This prevents a hidden dependence on latent dimensionality.

## 3. Contracts and example

Preserve floating dtype/device and do not mutate inputs. Reject incompatible shapes/dtypes/devices, empty tensors, non-floating tensors, non-finite tensor values, invalid alpha or invalid beta with `ValueError`. CPU float32 and float64 are tested.

If current predictions equal their references, every margin is zero and the loss is $\log 2$. For one scalar pair with shared noise zero, current predictions `(0, 2)`, references `(1, 1)` and `beta=2`, the margin is $4$ and the loss is $\log(1+e^{-4})$.

## 4. Verification

T0 checks contracts and input preservation. T1 compares against scalar arithmetic, schedule endpoints and extreme margins. T2 verifies analytic gradients, frozen context and feature/batch duplication invariance. T3 trains a two-parameter affine noise predictor for 80 updates on four fixed preference pairs. Its loss must fall and each reference-relative preference margin must become positive.

## 5. Experiment and scope

The experiment uses synthetic scalar latents and actual gradient descent through your functions. It does not load an image model or a preference dataset. Increasing the preference margin does not require both absolute denoising errors to decrease: the objective controls their reference-relative difference. The charts therefore report the logistic loss and pairwise margins.

## 6. Run

```bash
paperleet init diffusion_rl/diffusion_dpo
paperleet test diffusion_rl/diffusion_dpo
```

## 7. Sources

- Wallace et al., [paper, Section 4](https://arxiv.org/html/2311.12908v1).
- [Official `train.py`](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py): shared pair noise/timestep and DPO loss computation.

Code here is an original compact teaching implementation of those equations, not a vendored training script.
