# SimSiam implementation notes

- **Stop the target.** Detaching changes derivatives without changing forward values. Loss-value agreement cannot reveal accidental target training.
- **Keep prediction paths live.** A projection still learns through a prediction computed from it. Stopping its use as a target must not detach the shared upstream prediction path.
- **Cross the views.** Predict the second view from the first and vice versa. Asymmetric fixtures distinguish this from matching each prediction to its own view.
- **Sign and reduction.** Aligned directions give `-1`. Sum features, average examples, and average both directions. Averaging feature products introduces an extra factor of `1/D`.
- **Normalize along features.** Each example has its own norm (`dim=1`). Batch-axis normalization incorrectly couples examples.
- **Epsilon convention.** This exercise fixes a `1e-12` norm floor. A zero vector can have zero forward loss and a nonzero derivative. Other epsilon choices can be valid elsewhere but differ from this contract.

Nine mutants cover target training, detached predictions, wrong sign, omitted prediction normalization, wrong normalization axis, same-view matching, summed directions, one-direction training, and feature averaging. The supplied training isolates predictions learning fixed targets; successful alignment is not evidence that an entire self-supervised model cannot collapse.
