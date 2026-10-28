# RT-1: Solution notes

Transpose logits to (B,K,L), normalize the last axis, and multiply the original features. Each learned token is a spatial convex combination; normalizing K or pooling normalized features changes the operator.

Transpose logits to (B,K,L), normalize the last axis, and multiply the original features. Each learned token is a spatial convex combination; normalizing K or pooling normalized features changes the operator.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Softmax normalizes learned tokens rather than spatial positions.
- The learned spatial weights are discarded.

## Experimental scope

Isolate visual-token compression with supplied MLP attention logits; omit EfficientNet, the Transformer and physical robots.

## Reproducible protocol

Three spatial patches, one token and target pooled value .8; zero scores, 100 gradient steps at lr 1. Require MSE below .1% of its initial value.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
