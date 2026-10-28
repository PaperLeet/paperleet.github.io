# π₀.₅: Solution notes

Prefix queries see prefix only. FAST queries see prefix and causal FAST tokens. Expert queries see prefix and all expert tokens. FAST carries clean action labels, so expert-to-FAST access leaks the flow-matching target. Mask padding on both query and key axes.

Prefix queries see prefix only. FAST queries see prefix and causal FAST tokens. Expert queries see prefix and all expert tokens. FAST carries clean action labels, so expert-to-FAST access leaks the flow-matching target. Mask padding on both query and key axes.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Action-expert queries can read clean FAST action labels.
- The continuous action expert incorrectly uses causal rather than bidirectional attention.

## Experimental scope

Implement Appendix E / Figure 18 training attention connectivity. Prefix, discrete FAST labels and continuous action-expert contents are supplied; omit model pretraining.

## Reproducible protocol

Eight tokens split into three prefix, two FAST and three expert tokens; uniform legal attention and FAST perturbations 0/1/10/100/1000. Expert response must be zero, while prefix response exceeds .5. This is a forward information-flow experiment without a trainer.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
