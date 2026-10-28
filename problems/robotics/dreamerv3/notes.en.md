# DreamerV3: Solution notes

Apply symlog to raw scalars, then interpolate mass between adjacent centers. Encoded expectation equals the clipped transformed value. Decode the expectation in transformed space with symexp; averaging symexp of centers is a different operation.

Apply symlog to raw scalars, then interpolate mass between adjacent centers. Encoded expectation equals the clipped transformed value. Decode the expectation in transformed space with symexp; averaging symexp of centers is a different operation.

Identify reduction axes, validity masks and gradient paths from the equations before comparing a small hand calculation. Test oracles do not call the reference implementation.

## Mistakes caught by the tests

- Negative rewards lose their sign under symlog.
- Two-hot weights assume uniform spacing even for a nonuniform support.

## Experimental scope

Implement scale-robust scalar transforms and discrete-support encoding; omit the RSSM, imagined policy and Minecraft. two_hot receives values already in symlog space and caller-supplied centers.

## Reproducible protocol

Six scalars spanning magnitudes and 25 centers over [-6,6]; zero logits and 150 cross-entropy gradient steps at lr 3. Transform round-trip error <1e-9, cross entropy below 40% of initial and final target KL <.1.

The judge fixes seed 42. All calculations use CPU float64, explicit inputs and plain gradient updates where applicable; no datasets or model weights are downloaded. Native PyTorch and actual Pyodide/NumPy-WASM run the same T0–T3 suite. Thresholds refer to this teaching adaptation.
