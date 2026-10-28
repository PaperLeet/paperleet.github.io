# Solution reasoning and pitfalls

Traverse the schedule in execution order, recover a clean estimate and form the next noise level. Run the prefix without gradients and detach the boundary once; preserve the complete chain rule within the suffix.

$$
\hat x_0^{(i)}=(x_i-\sigma_i\epsilon_\theta(x_i,i))/\alpha_i,\quad x_{i+1}=\alpha_{i+1}\hat x_0^{(i)}+\sigma_{i+1}\epsilon_\theta(x_i,i).
$$

Initial noise and schedules are frozen. Prefix transitions execute without gradients; suffix transitions retain all recurrent state and model paths.

Detaching every step removes earlier suffix gradients. Truncation must change derivatives without changing forward samples. Schedule values are amplitudes, not variances.

Test basis: Hand-computed constant-epsilon chains, forward equality across truncations, independent scalar derivative recurrence and zero prefix gradients.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.05 (45 updates).

```json
{"final": 4.760992065567561e-13, "initial": 1.0, "updates": 45}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
