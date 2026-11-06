# Implementation notes

Read these after attempting the exercise.

Normalize feature rows with a norm floor, broadcast feature-center differences and reduce over the feature axis, then take `argmin` over centers. Cluster assignment is a detached, discrete target. Classification gradients flow through all logits; normalization remains differentiable when used in the encoder path. Use stable log-sum-exp for mean cross entropy.

All floating matrices are finite float32/float64, nonempty, with matching feature widths, dtype and device. Labels are int64 and in range. Accept strided inputs and leave them intact. Zero rows normalize to zero; the norm floor is `eps=1e-12`. `assign_clusters` accepts features already normalized by the caller and does not renormalize centers. Loss is a mean over supplied examples: the original training sampler balances clusters; this function does not silently insert class weights. Provided validation remains in the starter.

The maintained mutations isolate these faults:

- Normalize columns instead of feature rows.
- Choose the farthest cluster.
- Replace Euclidean assignment with unnormalized dot products.
- Sum pseudo-label losses across examples.
- Treat every cluster label as zero.
- Stop gradients through feature normalization.

## Actual CPU calibration

The lab alternates cluster refresh and real learning on nine deterministic 2D points. A 2×2 encoder and 2×3 head have ten trainable entries. It performs 90 SGD updates at learning rate `.25`, refreshing assignments and mean centers at updates 0, 30, 60 and 90. Empty toy centers retain their previous values. The supplied experiment calls all three submitted functions. T3 requires all three clusters occupied, encoder motion above `.01`, pseudo-label accuracy 1 and final loss below `.1`. These are synthetic training metrics, without fitting PCA or training an image encoder.

All data and initial parameters are deterministic; the test fixture also seeds Python/NumPy/PyTorch to 42. Native calibration uses the repository's PyTorch 2.14.1 CPU environment on Apple silicon. Tests use the same full training loop in the NumPy autograd adapter and exported WASM bundle. The gates validate numerical behavior and learning on this synthetic task, not a published paper benchmark.

Source: [paper](https://arxiv.org/abs/1807.05520) and [official implementation](https://github.com/facebookresearch/deepcluster/blob/main/clustering.py).

Observed native reference metrics: initial_loss=1.09861; final_loss=0.0695958; initial_accuracy=0.333333; final_accuracy=1.

Nonzero rows shorter than eps belong to the linear branch z=x/eps. The strided parent VJP there is g/eps, with zero gradients in unused storage. Handling only exactly zero rows misses this boundary.
