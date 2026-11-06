# Implementation notes

Read these after attempting the exercise.

Use transpose and flips on the two spatial axes, concatenate the 0°, 90°, 180°, 270° blocks, and repeat each label B times. Preserve the image autograd path: rotation is a pixel permutation. For the loss, gather the supplied class column, subtract it from stable row log-sum-exp, and average examples.

Images are nonempty finite float32/float64 `[B,C,H,H]`; square shapes allow all views to share a tensor. Output is `[4B,C,H,H]` plus int64 `[4B]` labels on the same device. Rotation-major batching is this exercise’s convention; the official data loader first creates views per sample. There is no interpolation, crop, color transform or input mutation. Loss accepts any nonempty `[M,4]` logits and int64 `[M]` labels in `[0,3]`, even when M is not a multiple of four. Strided images work. A 1×1 image still gets all four labels although those views are indistinguishable.

The maintained mutations isolate these faults:

- Swap clockwise/counterclockwise 90-degree views.
- Use sample-major labels for rotation-major views.
- Leave the 180-degree view unrotated.
- Detach augmented views from input images.
- Sum instead of averaging cross entropy.
- Exponentiate unshifted logits.

## Actual CPU calibration

Eight deterministic asymmetric 4×4 glyphs generate 32 rotated examples. A zero-initialized 16×4 linear head learns for 80 full-batch SGD updates at learning rate `.6`. Both submitted functions are used. T3 requires rotation accuracy 1 and loss below `.06` and 5% of its initial value. Symmetric images cannot always reveal their rotation; the synthetic glyphs are chosen to make this pretext task learnable. Rotation accuracy alone is not evidence of transfer representation quality.

All data and initial parameters are deterministic; the test fixture also seeds Python/NumPy/PyTorch to 42. Native calibration uses the repository's PyTorch 2.14.1 CPU environment on Apple silicon. Tests use the same full training loop in the NumPy autograd adapter and exported WASM bundle. The gates validate numerical behavior and learning on this synthetic task, not a published paper benchmark.

Source: [paper](https://arxiv.org/abs/1803.07728) and [official implementation](https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py).

Observed native reference metrics: initial_loss=1.38629; final_loss=0.0371615; initial_accuracy=0.25; final_accuracy=1.

Rotation affects only spatial axes. Every batch member and channel must survive, including when batch, channel and spatial dimensions are strided views. A scalar coordinate oracle checks the resulting views and parent-storage gradients.
