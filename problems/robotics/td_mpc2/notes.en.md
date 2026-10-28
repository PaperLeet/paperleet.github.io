# TD-MPC2: Solution notes

SimNorm normalizes each group separately. Elite weights normalize E and are shared across the trajectory horizon. Official temperature multiplies values; compute variance around the weighted mean and clamp standard deviation afterward. Use exact normalization, omitting the source denominator epsilon 1e-9.

SimNorm normalizes each group separately. Elite weights normalize E and are shared across the trajectory horizon. Official temperature multiplies values; compute variance around the weighted mean and clamp standard deviation afterward. Use exact normalization, omitting the source denominator epsilon 1e-9.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Temperature divides elite values instead of multiplying them.
- Second moment is mistaken for centered variance.

## Experimental scope

Isolate SimNorm and MPPI weighted moments for already-selected elites. Candidate trajectories and values are supplied; omit learned dynamics/Q networks and real MPC.

## Reproducible protocol

Two-step scalar actions, a 25-point two-dimensional grid, eight elites and twelve updates; temperature 5, std range [.03,1]. Start at zero mean and require final error below 3% of initial; also check mass sums of two three-dimensional simplexes.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
