# Diffusion Policy: Action Denoising and Receding-Horizon Execution

## 1. Exercise goal

Build three pieces of [Diffusion Policy](https://arxiv.org/abs/2303.04137): forward noise on action sequences, epsilon prediction loss, and the action chunk executed before replanning. The observation encoder and conditional prediction network are supplied by the caller.

## 2. Interfaces and returns

| Function | Inputs | Return |
|---|---|---|
| `add_action_noise(actions, noise, alpha_bar)` | Matching `(B,T,D)` normalized actions/noise; `(B,)` cumulative alphas | Noisy actions, same shape/dtype/device |
| `action_denoising_loss(prediction, noise)` | Matching `(B,T,D)` predicted and sampled epsilon | Scalar Tensor, same dtype/device |
| `action_chunk(actions, observation_horizon, execution_horizon)` | `(B,T,D)` planned actions; integer `To`, `Ta` | `(B,Ta,D)` execution slice |

All tensors are finite CPU float32/float64, batch/horizon/action dimensions are nonempty, and inputs remain unchanged. Validation helpers are provided: `_sequence` and `_matching` return `None` or raise `ValueError`; they do not compute exercise outputs.

### Return requirements

`add_action_noise` applies one cumulative alpha to every action coordinate in each example. It preserves differentiation through actions, noise and alpha. `action_denoising_loss` averages over **all** coordinates and detaches the target noise, while keeping prediction gradients. `action_chunk` preserves differentiation through selected actions; returning a view is allowed, modifying inputs is not.

## 3. Equations

With the usual DDPM schedule notation, where `alpha_bar` is the cumulative alpha product rather than a per-step alpha:

$$
A^k=\sqrt{\bar\alpha_k}A^0+\sqrt{1-\bar\alpha_k}\epsilon,\qquad L=\frac{1}{BTD}\sum_{b,t,d}(\hat\epsilon_{btd}-\operatorname{stopgrad}(\epsilon_{btd}))^2.
$$

Require `0 < alpha_bar[b] < 1`. Observations condition the denoiser; they are not part of the noised action tensor in this exercise.

The prediction window contains past-aligned actions. Under the official code's `oa_step_convention=True`, its current-time action is at index `To-1`. Execute:

```python
start = observation_horizon - 1
chunk = actions[:, start:start + execution_horizon, :]
```

Require positive integer horizons and `To-1+Ta <= T`. Replan after this chunk; do not execute all `T` predictions automatically.

## 4. Worked example

An action `2`, noise `-1`, and cumulative alpha `.64` produce `.8*2 + .6*(-1) = 1`. A sequence `[10,20,30,40,50]` with `To=2`, `Ta=3` executes `[20,30,40]`.

## 5. Deterministic experiment

The supplied CPU experiment uses a scalar goal and a known smooth time profile to define 2D demonstration actions. A two-coefficient predictor learns noise from noisy actions plus that goal-conditioned profile at one noise level. Your functions create the noisy batch, train the predictor and extract the execution chunk. Charts compare predicted actions with demonstrations and a removed-conditioning baseline.

This is a teaching adaptation of conditional action diffusion. It isolates training and execution alignment; it does not train an image encoder, run a multistep reverse sampler, or reproduce a manipulation benchmark.

## 6. Run your implementation

Implement the three functions in `starter.py`, then run the package through PaperLeet. T0 checks contracts, T1 independent numerical examples, T2 gradients and invariances, and T3 the conditional denoising experiment.

## 7. Sources

- [Paper and official project](https://diffusion-policy.cs.columbia.edu/): RSS 2023, Section 2.3.
- [Official low-dimensional policy](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py): `compute_loss`, `predict_action`, and `oa_step_convention`.
