# Solution reasoning and pitfalls

Replicate one initial noise vector across each prompt group. Center and scale each reward channel within its own group, then sum channel advantages. Broadcast terminal advantages to transitions and apply the clipped objective.

$$
A_{pg}=\sum_k\frac{R_{pgk}-\mu_{pk}}{\sqrt{\operatorname{mean}_g(R_{pgk}-\mu_{pk})^2}+\varepsilon},\quad L=-\operatorname{mean}_{p,g,t}\min(e^{l_{pgt}}A_{pg},\operatorname{clip}(e^{l_{pgt}},1-c,1+c)A_{pg}).
$$

Noise and reward-derived advantages are frozen. Only current log ratios differentiate in the policy loss; each prompt is normalized independently.

Do not normalize across prompts or combine differently scaled rewards before normalization. A constant reward channel contributes zero. Shared initial noise does not require sharing later SDE draws.

Test basis: Hand-computed two-channel advantages, prompt isolation, constant channels, shared-noise copies, signed clipping and frozen gradients.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.03 (70 updates).

```json
{"final_error": 0.00024607032777813887, "initial_error": 1.0, "prompt_groups": 2}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
