# Solution reasoning and pitfalls

Measure policy movement with exact equal-covariance Gaussian KL. Block only when advantage and log-ratio signs agree and divergence strictly exceeds the threshold. Replace blocked logits before exponentiation to avoid zero-times-infinity; retain reference KL on every step.

$$
D=\|\mu_\theta-\mu_{old}\|^2/(2s^2),\quad M=1-\mathbf1[D>\delta\ \land\ ((A>0\land l>0)\lor(A<0\land l<0))],\quad L=\operatorname{mean}[-M e^l A+\beta D_{ref}].
$$

Only log_ratios and reference_kl differentiate in loss. Advantages and divergence decisions are frozen. The mask is zero only for favorable-direction updates beyond the divergence threshold; blocked ratios are replaced before exponentiation.

Large divergence alone must not block corrective updates. Equality at D=delta or ratio=1 remains active. Mask decisions are nondifferentiable, while reference-KL gradients remain active even on blocked policy terms.

Test basis: Four directional cases, strict threshold boundaries, blocked extreme ratios, analytic gradients and online Gaussian training.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.06 (65 updates).

```json
{"final_error": 3.293515663401782e-06, "initial_error": 0.8233882030178326, "updates": 65}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
