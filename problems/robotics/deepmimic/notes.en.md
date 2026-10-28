# DeepMimic: Solution notes

Within each component, sum squared joint/end-effector coordinates before applying its paper-specific exponential scale. The four weights sum to one, so perfect matching yields reward one. Equal end-effector errors are penalized more sharply than angular-velocity errors.

Within each component, sum squared joint/end-effector coordinates before applying its paper-specific exponential scale. The four weights sum to one, so perfect matching yields reward one. Equal end-effector errors are penalized more sharply than angular-velocity errors.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- End-effector position errors use the angular-velocity scale.
- Pose error averages joints instead of summing them.

## Experimental scope

Implement pose, angular-velocity, end-effector and center-of-mass imitation rewards. Pose inputs are precomputed quaternion geodesic angle errors; omit PPO, pose parsing and physical simulation.

## Reproducible protocol

31 motion candidates with error scale from one to zero, two joints and two end effectors; evaluate shaped rewards without training. Require perfect-match reward one, initial reward below .75, 30 strict improvements and all rewards in (0,1].

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
