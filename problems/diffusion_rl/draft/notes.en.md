# Solution reasoning and pitfalls

Retain the original generated sample’s reward gradient. Freeze that sample when constructing additional noisy inputs, reconstruct with the trainable denoiser and evaluate rewards. Average over the batch and then over original plus extra branches.

$$
z_k=\alpha\operatorname{sg}(x_0)+\sigma\operatorname{sg}(\epsilon_k),\quad \hat x_k=(z_k-\sigma\epsilon_\theta(z_k))/\alpha,\quad L=-\frac{r(x_0)+\sum_{k=1}^K r(\hat x_k)}{K+1}\text{ averaged over batch}.
$$

Original generated samples retain their reward gradient. Re-noising uses generated.detach() and frozen noises; additional gradients reach only the denoiser. Reward parameters are supplied/frozen by caller but input gradients remain live.

Without detaching the re-noising anchor, the initial sampling chain receives unintended repeated gradients. Detach inputs, not reconstructions. Averaging follows the text’s gradient averaging and differs from pseudocode summation by a fixed learning-rate factor.

Test basis: Hand-evaluated constant-denoiser objective, original gradient scaled by 1/(K+1), denoiser gradients and noise-order invariance.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.08 (55 updates).

```json
{"final": 0.07564035181237105, "initial": 1.015, "weight": 0.6896500813549452}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
