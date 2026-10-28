# Solution reasoning and pitfalls

Equal-covariance Gaussian KL reduces to a variance-weighted squared mean distance. Sum over time for the trajectory bound, average over samples, and combine the frozen-reward score term with differentiable KL.

$$
K_{bt}=\frac{\|\mu_{bt}-\mu^{ref}_{bt}\|^2}{2s_t^2},\quad L=\frac1B\sum_b[-\alpha\operatorname{sg}(r_b)\sum_t\ell_{bt}+\beta\sum_t K_{bt}].
$$

Current Gaussian means and KL penalties differentiate. Reference means, scales and terminal rewards are frozen; supplied logp differentiates. State-visitation gradients omitted as in Eq. (9).

The closed-form KL is a distributional distance, not a sampled log ratio. Do not average across time or detach the KL penalty.

Test basis: Scalar KL, linear scaling under timestep repetition, analytic reward/regularizer gradients, zero-KL endpoints and a regularized online bandit.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.035 (70 updates).

```json
{"final_error": 0.00024093245881487226, "initial_error": 0.6944444444444445, "mean": 0.8178113341446054}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
