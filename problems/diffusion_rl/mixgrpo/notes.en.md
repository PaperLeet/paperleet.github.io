# Solution reasoning and pitfalls

Advance the window by completed shift intervals and clamp it at the final valid position. ODE steps use velocity integration; SDE steps add score correction and a nonzero scale. Slice the optimized window before averaging the clipped objective.

$$
l=\min(\lfloor m/\tau\rfloor s,T-w),\quad W=\{l,\ldots,l+w-1\}.\quad L=-\operatorname{mean}_{b,t\in W}\min(e^{\ell_{bt}}A_b,\operatorname{clip}(e^{\ell_{bt}},1-c,1+c)A_b).
$$

Rollout states are detached, velocity remains differentiable. Advantages are frozen; loss gradients are zero outside the selected window.

The window saturates rather than wrapping around. Outside steps still contribute to the final rollout but not to the policy loss; dividing by total steps dilutes gradients. Sampling time decreases, so dt is negative.

Test basis: Window boundary cases, analytic ODE/SDE moments, zero unselected gradients, selected-window reduction and online mixed sampling.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.15 (60 updates).

```json
{"early_error": 0.8417650037927409, "last_start": 2, "late_error": 0.0029253598150183547}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
