# BYOL implementation notes

1. The official implementation sums directions before averaging the batch. There is no additional `1/2`; copying SimSiam's reduction changes the scale.
2. The official helper uses `sqrt(max(sum(x*x),1e-12))`. Using `1e-12` as a norm floor changes near-zero vectors.
3. Match predictions to the opposite view's target, and keep regression gradients on predictions only.
4. EMA's large weight belongs to the old target. Target movement comes from parameter averaging, not loss.backward.
5. Even at `m=0/1`, return detached independent storage so later result edits cannot mutate inputs.
6. Do not normalize EMA parameters or average unrelated predictor and target parameters.

Native reference calibration: 80 SGD/EMA updates, 45 online parameters, loss 2.035516 → 0.002384, total target movement about 7.568853. Scalar loss/EMA errors are approximately 2.22e-16. Environment: macOS arm64, Python 3.14.2, PyTorch 2.14.1, CPU float64. Nine mutants cover directional scale, target gradients, pairing, epsilon, feature reduction, EMA weights, EMA gradients and omitted normalization.

The gradient suite also checks interleaved strided parent tensors, including zero gradients in unused coordinates.
