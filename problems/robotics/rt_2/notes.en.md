# RT-2: Solution notes

Construct the allowed set per row, replace disallowed logits with negative infinity, then apply stable softmax. Zeroing probabilities afterward breaks normalization. Disallowed robot logits have zero output probability and gradient.

Construct the allowed set per row, replace disallowed logits with negative infinity, then apply stable softmax. Zeroing probabilities afterward breaks normalization. Disallowed robot logits have zero output probability and gradient.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Robot rows retain forbidden language tokens.
- The action-only vocabulary is incorrectly applied to vision-language rows.

## Experimental scope

Implement output constraints for robot and vision-language rows sharing logits; callers supply action-token identities. Omit PaLI-X, PaLM-E and semantic reasoning.

## Reproducible protocol

Two task rows, four tokens, two legal action tokens; zero logits and 80 negative-log-probability steps at lr 1. Require loss below 4% of initial, zero forbidden robot mass and positive VLM mass outside the action subset.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
