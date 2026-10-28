# DayDreamer: Solution notes

Use log_softmax for logq/logp. The dynamics term freezes all of q, both probabilities and logq; the representation term freezes logp. Alpha one updates only the prior, alpha zero only the posterior. A weighted undetached KL cannot reproduce this split.

Use log_softmax for logq/logp. The dynamics term freezes all of q, both probabilities and logq; the representation term freezes logp. Alpha one updates only the prior, alpha zero only the posterior. A weighted undetached KL cannot reproduce this split.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Dynamics KL wrongly updates the posterior encoder.
- Dynamics KL averages instead of summing independent categorical latents.

## Experimental scope

Isolate the DreamerV2-style KL balancing used by DayDreamer, with default alpha=0.8. Compute KL before free-nats clipping; omit the RSSM and online robot training.

## Reproducible protocol

One example, two latent variables, three classes each; fixed posterior and uniform initial prior, alpha .8, 80 gradient steps at lr 1. Require KL below 1e-5 of initial and maximum probability error below .002.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
