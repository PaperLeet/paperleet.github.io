# DiffMimic: Solution notes

Replay replaces excessive-error states before the next policy/physics transition. Kept states pass gradients; replayed branches cut gradients to old states. State-matching loss must still measure actual trajectory errors instead of erasing an error because replay occurs. The PD target angular velocity is zero.

Replay replaces excessive-error states before the next policy/physics transition. Kept states pass gradients; replayed branches cut gradients to old states. State-matching loss must still measure actual trajectory errors instead of erasing an error because replay occurs. The PD target angular velocity is zero.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- States exactly at the threshold are kept instead of replayed.
- PD control omits velocity damping.
- State matching averages the time horizon instead of summing it.

## Experimental scope

Implement error-triggered demonstration replay, PD torque and temporal state matching. The experiment uses a scalar differentiable integrator, not a Brax humanoid reproduction.

## Reproducible protocol

Three starts, twelve scalar-integrator steps, dt .15, PD gains (2,.4), replay threshold 2; policy gain starts at one and takes 60 gradient steps at lr .04. Require loss below 90% of initial, negative final gain and a separate replay probe at threshold .1.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
