# ACT: Solution notes

The official implementation applies a negative exponential to oldest-to-newest indices, favoring older predictions for positive decay. Explicit validity replaces its all-zero sentinel so zero actions remain valid. Logvar means log variance.

Padding must not consume exponential ranks, and all action coordinates share temporal weights. Averaging instead of summing latent dimensions changes KL strength with latent size.

## Mistakes caught by the tests

- Temporal weights favor newest predictions instead of the declared official ordering.
- KL incorrectly averages latent dimensions.

## Experimental scope

Implement two ACT tensor cores with predictions already aligned to one execution time; omit the visual Transformer, action-chunk network and teleoperation system.

## Reproducible protocol

64 sinusoidal targets, five overlapping predictions, decay .01; latest prediction is the baseline. Fit a 4×3 style mean toward a standard normal using 40 gradient steps at lr .5. Require ensemble MSE below 10% of latest-prediction MSE and final KL below 1e-4 of initial KL.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
