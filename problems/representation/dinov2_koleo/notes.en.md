# Implementation notes

Read these after attempting the exercise.

Normalize rows, compute the detached pairwise cosine matrix, exclude diagonal entries, and select each row’s maximum. Gather neighbors from the differentiable normalized tensor, compute distances with a fixed `1e-8` coordinate shift and average negative log distances. Detaching the gathered vectors loses valid neighbor gradients. The negative sign makes minimizing the loss spread close points.

Use finite float32/float64 `[B,D]`, B≥2 and D≥1; default `eps=1e-8` must be finite and positive. `nearest_neighbors` accepts already normalized rows and returns int64 IDs. Strict self-exclusion and lowest-index ties are explicit boundary conventions; the official code masks the diagonal with -1, which can admit self in an exact antipodal tie. This exercise masks it with negative infinity. Preserve input data, floating dtype/device and strided gradient paths. Zero rows normalize to zero. At neighbor-switch boundaries the argmax is nondifferentiable; gradient tests use unique neighbors.

The maintained mutations isolate these faults:

- Include the anchor itself as a neighbor.
- Select farthest non-self vectors.
- Compute neighbors and distances on raw features.
- Detach selected neighbors and lose their endpoint gradients.
- Minimize positive log distance.
- Omit the official PairwiseDistance coordinate epsilon.
- Sum regularization over the batch.

## Actual CPU calibration

Six three-dimensional vectors start close together and undergo 100 SGD updates at learning rate `.06` using the submitted loss. There are 18 trainable entries. The plot records actual loss and an independently measured mean nearest-neighbor Euclidean distance. T3 requires final distance above 1 and more than five times its initial value, with final loss below zero. This demonstrates KoLeo alone; it neither trains a DINOv2 image encoder nor establishes semantic quality or collapse prevention for an arbitrary network.

All data and initial parameters are deterministic; the test fixture also seeds Python/NumPy/PyTorch to 42. Native calibration uses the repository's PyTorch 2.14.1 CPU environment on Apple silicon. Tests use the same full training loop in the NumPy autograd adapter and exported WASM bundle. The gates validate numerical behavior and learning on this synthetic task, not a published paper benchmark.

Source: [paper](https://arxiv.org/abs/2304.07193) and [official implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py).

Observed native reference metrics: initial_loss=2.31616; final_loss=-0.109539; initial_nearest_distance=0.103081; final_nearest_distance=1.11579.

Nonzero vectors shorter than `eps` divide by that caller-supplied norm floor. Independent scalar and finite-difference checks cover this linear normalization region. The fixed-inner-epsilon convention matches the official code for nondefault `eps` as well.
