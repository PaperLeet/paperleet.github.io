# DAgger: Solution notes

Select one policy action per state rather than averaging action vectors. Concatenate on the sample axis and retain old labels. Expert labels belong to states visited by the learner; they are not the executed mixture actions.

Beta zero uses only the learner and beta one only the expert. Explicit uniform draws make selection reproducible. The dataset does not retain previous policy graphs.

## Mistakes caught by the tests

- A policy mixture incorrectly averages action vectors.
- Aggregation discards every previous round.

## Experimental scope

Isolate Algorithm 3.1 in a scalar linear environment with policy refitting; no interactive human expert or full classifier.

## Reproducible protocol

Eight starts, eight steps per round, four rounds, two initial zero examples; least-squares feedback fit with explicit draws. Gain starts at zero. Require final twelve-step state energy below 5% of baseline, 258 examples, and gain -0.6.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
