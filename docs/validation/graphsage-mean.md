# GraphSAGE mean aggregation

## Scope and sources

Module `representation/graphsage_mean` (#93), paper `graphsage` (#61).

Checked 2026-10-16 against the [paper](https://arxiv.org/pdf/1706.02216), [project](https://snap.stanford.edu/graphsage/) and [author repository](https://github.com/williamleif/GraphSAGE). The exercise independently implements Algorithm 1 lines 4–7 with the Section 3.3 mean operator, row-vector weights, ReLU and per-hop feature-axis L2 normalization. It does not claim to implement Equation (2)'s self-inclusive GCN variant or reproduce all released TensorFlow choices. Padding, empty neighborhoods and epsilon behavior are explicit teaching adaptations.

Author code was inspected read-only: `graphsage/aggregators.py` blob `7dbd2523f753125b31e078f25b6edd7949b945c7` is not an exercise dependency. The published repository is linked for comparing configurable self/neighbor projection, concatenation and final activation choices; the independent tests use a scalar Python oracle rather than importing the reference.

## Numerical and training checks

Eleven tests cover float32/64, int32/64 IDs, strided tensors, invalid padding sanitized before indexing, repeated samples, zero slots/nodes/depth, validation and input preservation. Hand-computed values, scalar multi-hop calculations, analytical repeated-index gradients and 60 scalar finite differences independently check the implementation. Additional checks cover node/sample permutations, per-hop normalization, zero ReLU and epsilon boundaries.

The supplied synthetic training experiment performs 100 actual Adam updates on 80 shared parameters, trains on a 14-node graph, then freezes weights for a distinct 18-node graph. Supervised queries have zero features and balanced labels, so a feature-only classifier has a 50% ceiling. Reference training loss decreases from 0.693147 to approximately 0.006113; both graph query accuracies reach 1.0. These are teaching results, not paper dataset metrics.

Initial native calibration: all 11 tests pass, all 8 mutants are caught consistently over three repetitions (297 cases, 3.86 seconds total). Mutants cover sum/mean, padding counts, unsafe masked IDs, missing own features, wrong normalization axis, final-only normalization, omitted ReLU and detached neighbor gradients. The T3 budget is eight seconds; no existing tolerances or budgets are relaxed.

Initial exact exported WASM calibration (StrideTorch 0.31, Pyodide 0.29.4, NumPy 2.2.5): all 11 tests pass and all 8 mutants are caught. VM preparation 1,671 ms, calibration 1,867 ms, subsequent full learner suite 149 ms. These are local harness measurements with runtime packages already available; they exclude first-download/network time and are not a controlled browser/server speed comparison.

## Release validation

The feature must also pass the current main branch's complete native, infrastructure, frontend and exported WASM gates before publication. A successful quality check alone does not establish an actual Pages or Oracle deployment.
