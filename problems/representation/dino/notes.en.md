# DINO implementation notes

1. Teacher probabilities use (raw_teacher − center) / teacher_temperature. Subtracting the center after softmax or after division is a different expression.
2. Detach both teacher and center. Student log probabilities remain differentiable; sharp teacher probabilities must still sum to one.
3. The first two student views match teacher global views. Exclude only t=v, and include every additional student crop for both teachers. The number of pairs is 2*(V−1).
4. Average pairs and examples, sum classes. The same-view exclusion changes the analytic target mixture for each student view.
5. Update the center from raw teacher logits averaged over both views and examples. The momentum coefficient belongs to the previous center.
6. Compute the current loss using the old center, then install the detached new center. Center EMA does not replace the teacher-network parameter EMA used in full DINO.

Native reference calibration: 80 SGD/center updates, 54 parameters, cross-entropy 1.107223 → 0.024133, synthetic class agreement 1.0. Scalar errors are below 3e-16; teacher gradients remain absent. Environment: macOS arm64, Python 3.14.2, PyTorch 2.14.1, CPU float64. Eleven mutants cover centering, teacher sharpening, teacher gradients, view exclusions, pair reduction, student temperature, the center's view axis, momentum weighting and center gradients.

The gradient suite also checks interleaved strided parent tensors, including zero gradients in unused coordinates. The multi-view scalar and analytic-gradient cases include five student views, so extra local crops and the 2*(V−1) reduction are exercised.
