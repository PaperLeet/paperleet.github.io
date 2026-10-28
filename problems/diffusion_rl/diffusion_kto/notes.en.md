# Solution reasoning and pitfalls

Average unrelated log ratios and clamp the baseline at zero. A desirable label retains the relative-policy direction; an undesirable label reverses it. Center sigmoid utility at zero and negate it for gradient descent.

$$
b=\max(0,\operatorname{mean}\operatorname{sg}(\ell_{unrelated})),\quad z=w\beta(\ell_\theta-\operatorname{sg}(\ell_{ref})-b),\quad L=-\operatorname{mean}(\sigma(z)-1/2).
$$

Only current log probabilities receive gradients. Reference, feedback labels and the unrelated-batch baseline are detached.

The baseline comes from unrelated pairs, not the current feedback batch. Beta scales the log ratio and the baseline together. This is sigmoid utility, not log-sigmoid classification.

Test basis: Per-example scalar sigmoid, clamped baselines, label symmetry, analytic derivatives and frozen inputs.

## Experiment calibration

Seed 42, CPU float64, direct gradient descent, no downloaded data. Values below were measured from the reference, not a preset trajectory. PyTorch 2.14.1. lr=.2 (60 updates).

```json
{"final": -0.4760557262293378, "initial": -0.0, "margin": 1.8632436167371014}
```

Acceptance thresholds are in `tests/test_core.py::test_real_toy_experiment`; the baseline is the measured output before training.
