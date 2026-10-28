# Solution reasoning and pitfalls

Form a rejected-minus-preferred score margin and apply stable softplus. Detach the already sampled latent before solving the noising equation for the clean sample. Freezing reward parameters must preserve the derivative with respect to its input.

$$
L_{rank}=\operatorname{mean}\,\operatorname{softplus}(r_l-r_w),\qquad \hat x_0=(\operatorname{sg}(x_t)-\sigma_t\epsilon_\theta)/\alpha_t.
$$

Ranking differentiates both scores. ReFL stops the rollout latent and differentiates epsilon; the frozen reward model must still propagate gradients to its input.

Averaging margins before logistic would cancel opposing preferences. Alpha is an amplitude, so do not take another square root. The latent has no gradient and the epsilon derivative is -sigma/alpha.

Test basis: Scalar logistic oracle, extreme scores, noising inversion and analytic gradients; T3 trains a scorer then optimizes a generator parameter.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. ranking lr=.3 (40 updates); generator lr=.03 (30 updates).

```json
{"final": 0.089454537799872, "initial": 0.6931471805599453, "reward_gain": 1.9853489655118983}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
