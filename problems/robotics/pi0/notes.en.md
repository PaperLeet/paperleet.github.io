# π₀: Solution notes

Official code parameterizes time from actions to noise, so the target velocity is noise-actions. Negative sampling steps transport noise back to actions. Detach the supervised label, not the interpolated state.

Official code parameterizes time from actions to noise, so the target velocity is noise-actions. Negative sampling steps transport noise back to actions. Detach the supervised label, not the interpolated state.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Flow labels use the opposite time direction.
- Reverse-time Euler changes the sign twice.

## Experimental scope

Isolate the official openpi linear flow path and Euler sampling direction; explicit noise/time and a linear velocity regressor replace the VLA action expert.

## Reproducible protocol

A 6×2×1 action block, noise equal to negative actions, six evenly spaced times; fit a zero-initialized scalar velocity gain for 60 gradient steps at lr .5, then integrate ten steps at dt -.1. Require velocity MSE below 1e-10 of initial and sample MSE below 1e-10.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
