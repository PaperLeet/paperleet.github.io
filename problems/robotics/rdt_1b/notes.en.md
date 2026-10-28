# RDT-1B: Solution notes

Do not compact empty slots: each slot retains its physical meaning. The mask distinguishes absent coordinates and clears invalid denoiser outputs before execution. The released training MSE is not a valid-coordinate masked loss, so this exercise does not substitute that objective.

A one-hot slot matrix preserves gradients to local actions. Reject duplicate slots rather than mixing physical meanings. A valid action equal to zero still has a present dimension.

## Mistakes caught by the tests

- The packing matrix uses local indices instead of semantic slots.
- Execution fails to mask absent action coordinates.

## Experimental scope

Isolate unified action placement and final sampling masks using a configurable small slot map, not the complete 128D schema or 1.2B Transformer.

## Reproducible protocol

Map three local coordinates into six unified coordinates; initialize predictions at nine and take 100 gradient steps at lr .5. Require loss below 1e-10 of initial loss and exactly zero unused execution coordinates.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
