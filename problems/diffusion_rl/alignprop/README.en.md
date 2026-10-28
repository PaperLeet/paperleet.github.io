# AlignProp Truncated Reward Backpropagation

## 1. Problem

Implement a DDIM sampling chain whose terminal reward differentiates through a selected denoising suffix.

Call a supplied small epsilon model to study time direction and gradient truncation; LoRA and activation checkpointing are outside scope.

## 2. Core equations

$$
\hat x_0^{(i)}=(x_i-\sigma_i\epsilon_\theta(x_i,i))/\alpha_i,\quad x_{i+1}=\alpha_{i+1}\hat x_0^{(i)}+\sigma_{i+1}\epsilon_\theta(x_i,i).
$$

## 3. From equations to code

```python
reward_rollout(initial: Tensor, alphas: Tensor, sigmas: Tensor, denoiser, backprop_from: int) -> Tensor
```

Traverse the schedule in execution order, recover a clean estimate and form the next noise level. Run the prefix without gradients and detach the boundary once; preserve the complete chain rule within the suffix.

## 4. Contract

initial (B,D); alphas/sigmas equal vectors of length T+1 in execution order noisy-to-clean, amplitudes alpha>0, sigma>=0. denoiser(x,step) returns same shape. backprop_from in [0,T-1] specifies first differentiable transition. Terminal schedule is caller-supplied.

Initial noise and schedules are frozen. Prefix transitions execute without gradients; suffix transitions retain all recurrent state and model paths.

## 5. Visualization and experiment

The experiment calls your submitted functions; plotted points are measured results.

A four-step DDIM chain with a scalar epsilon bias is trained for 45 updates on a quadratic terminal target; every rollout uses the submitted truncation routine.

## 6. Run

```bash
python -m paperleet.cli test diffusion_rl/alignprop --impl problems/diffusion_rl/alignprop/solution.py
python -m paperleet.cli mutate diffusion_rl/alignprop
```

## 7. Extensions

Measure gradient bias and memory requirements for different suffix lengths.

[Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) · Method: reward backpropagation; official sd_pipeline.py backprop_timestep
