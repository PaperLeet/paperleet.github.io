# Solution reasoning and pitfalls

Convert logits to stable log-sigmoid probabilities. Subtract adjacent log probabilities on each trajectory, multiply the winner-minus-loser margin by beta times T, then apply logistic loss.

$$
d_w=\log\sigma(a^w_{next})-\log\sigma(a^w_{now}),\quad d_l=\log\sigma(a^l_{next})-\log\sigma(a^l_{now}),\quad L=\operatorname{mean}\operatorname{softplus}(-\beta T(d_w-d_l)).
$$

All four classifier evaluations differentiate because both adjacent states contribute to each log probability ratio.

Raw logit differences are not log probability ratios. The derivative of log-sigmoid(a) is sigmoid(-a); both current-time terms and the total-step factor matter.

Test basis: Scalar probability-ratio oracle, four-branch analytic derivatives, timestep scaling and extreme logits.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.2 (50 updates).

```json
{"final": 0.05033276942721891, "initial": 0.6931471805599453, "weight": 1.7389346544025606}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
