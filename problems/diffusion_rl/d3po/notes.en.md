# Solution reasoning and pitfalls

Compute winner and loser log ratios relative to their references, then subtract at matching timesteps. Each transition has its own logistic objective; average over batch and time only afterward.

$$
z_{bt}=\beta[(\ell^w_{bt}-\ell^{w,ref}_{bt})-(\ell^l_{bt}-\ell^{l,ref}_{bt})],\qquad L=\frac1{BT}\sum_{bt}\operatorname{softplus}(-z_{bt}).
$$

Both current preferred and rejected log probabilities differentiate; both reference tensors are detached.

Summing across time before softplus changes the objective and cancels opposing timestep margins. Freeze the references, while retaining gradients through both current branches.

Test basis: Unequal per-step margins, timestep-duplication invariance, frozen-reference gradients and stable extreme margins.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.3 (60 updates).

```json
{"final": 0.0839873427692218, "initial": 0.6931471805599453, "minimum_margin": 1.368771445818979}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
