# Octo: Solution notes

Observation-window padding and action-horizon/dimension padding are distinct and must intersect. Divide by the valid fraction so adding padding does not dilute the loss; extremely sparse masks follow the source denominator floor.

Observation-window padding and action-horizon/dimension padding are distinct and must intersect. Divide by the valid fraction so adding padding does not dilute the loss; extremely sparse masks follow the source denominator floor.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Observation-window padding is ignored.
- Padding dilutes the loss because the valid fraction is omitted.

## Experimental scope

Isolate the action head masked_mean and intersection of timestep/action padding. Compute coordinatewise squared error before any head-specific action_dim scaling; omit the diffusion network.

## Reproducible protocol

A 1×2×2×2 action tensor with two valid coordinates and padded targets set to 100; 40 gradient steps at lr .2. Require loss below 1e-6 of initial and no movement in padded coordinates.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
