# Solution reasoning and pitfalls

An action consists of all feature coordinates, so sum log densities over features. Hold sampled actions and the old policy fixed. Multiply each transition ratio by the terminal advantage and choose the conservative clipped branch with minimum.

$$
\ell=-\frac12\sum_d[(a_d-\mu_d)^2/s^2+2\log s+\log(2\pi)],\quad L=-\operatorname{mean}_{b,t}\min(e^{\ell-\ell_{old}}A_b,\operatorname{clip}(e^{\ell-\ell_{old}},1-e,1+e)A_b).
$$

Gaussian actions and std are detached; current means differentiate. PPO old log probabilities and advantages are detached. Very large unfavorable log ratios may overflow; finite tested ratios stay within the declared numerical domain |current-old|<=40.

Do not average feature log densities or clip a product of full-trajectory ratios. For negative advantages the lower clip bound limits gains from excessively reducing action probability.

Test basis: Scalar Gaussian densities, signed clipping oracle, score gradients, frozen samples and timestep-duplication invariance.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.08 (60 updates).

```json
{"final_error": 1.7610391745268231e-06, "initial_error": 1.0, "updates": 60}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
