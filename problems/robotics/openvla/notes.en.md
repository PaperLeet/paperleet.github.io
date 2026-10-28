# OpenVLA: Solution notes

Encoding reverses IDs from the vocabulary end. The official np.digitize right=False convention assigns exact edges to the bin on their right. The maximum edge receives the last code and decoding clips it to the last center. Unnormalization transforms only masked coordinates.

Encoding reverses IDs from the vocabulary end. The official np.digitize right=False convention assigns exact edges to the bin on their right. The maximum edge receives the last code and decoding clips it to the last center. Unnormalization transforms only masked coordinates.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Exact edges use the wrong digitize boundary convention.
- Action IDs are not reversed from the vocabulary end.
- Unmasked coordinates are incorrectly converted to physical units.

## Experimental scope

Reproduce official ActionTokenizer conventions and action unnormalization without a 7B model. Smaller edge arrays make examples inspectable; the official default has 256 edges and 255 centers.

## Reproducible protocol

33 edges and 99 values form 33 three-dimensional actions spanning [-1.2,1.2]. Normalized quantization error <=1/32, physical error <=1/16, and more than 80 center-decoded codes round-trip identically; no trainer is used.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
