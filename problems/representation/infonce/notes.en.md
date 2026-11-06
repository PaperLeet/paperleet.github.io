# InfoNCE implementation notes

1. `positive_indices` contains candidate columns, not semantic category labels. Columns need not be zero.
2. Include all K candidates in each log-sum-exp, including the positive. Removing the positive breaks probability normalization and can produce negative losses.
3. Normalize over `dim=1`, then average over the batch. Duplicating the entire batch should preserve loss.
4. Raw `exp().sum().log()` overflows for large logits. Stable log-sum-exp can subtract each row's maximum first.
5. Keep the denominator differentiable. Its contribution includes every negative and the positive; the complete derivative is `(softmax-one_hot)/B`.
6. Scores are supplied directly. Additional temperature or normalization changes the contract. SimCLR's candidate construction and normalization belong to the next exercise.
7. With `K=1`, the sole candidate has probability one, so both loss and gradients vanish.

Calibration setup: deterministic trigonometric initialization; shared test seed 42; float64 `6×4` score table; 80 full-batch SGD updates at learning rate 1.5. Initial predictions provide the baseline. T3 requires final accuracy 1, mean positive probability >0.9, loss <0.1 and below 10% of its initial value, and minimum positive-negative margin >3. Maximum error against an independent scalar oracle must be <1e-12. Charts record actual updates, never a preset curve.

Native calibration: macOS 26.5.2 arm64, Python 3.14.2, PyTorch 2.14.1 CPU. Reference measurements: loss 1.527860 → 0.041209; accuracy 1/3 → 1; mean positive probability 0.959630; minimum margin 3.911801; scalar-oracle error 4.44e-16.
