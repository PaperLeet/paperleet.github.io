# BEiT implementation notes

The official training engine obtains target visual token IDs without gradients, selects the Boolean masked positions, and passes the selected logits and IDs to ordinary mean cross entropy. Targets are hard discrete IDs here. The image tokenizer, its training data, the masked-image Transformer and reconstruction of an image from tokenizer codes are outside the exercise.

Filtering happens before label gathering. This matters for visible sentinel IDs: they are not an `ignore_index` feature of the vocabulary criterion and must not accidentally index a class. Empty-mask zero is an explicit teaching extension; the normal BEiT training setup supplies masked tokens. A zero linked to `logits.sum()` gives the required all-zero gradient.

| Fault | Independent evidence |
| --- | --- |
| All-token or visible-token supervision | Visible score changes cannot affect the objective; masked gradients alone are nonzero. |
| Dividing by batch size | Unequal mask counts specify a token mean with a closed-form scalar answer. |
| Probabilities treated as logits | Scalar log-sum-exp and analytic `(p-one_hot)/M` gradients reveal the extra softmax. |
| Wrong target class | Nonuniform scores and nonconstant IDs pin each target coordinate. |
| Detached logits | The loss must have a live gradient with exact masked derivatives. |
| Wrong sign | Uniform scores give positive `log(K)`, and real training must improve CE. |
| Strided logit gradient loss | Interleaved parent VJPs preserve every selected class derivative and zero unused storage. |
| Gathering visible sentinel IDs | Mixed negative and positive out-of-range visible IDs must be ignored before gathering. |

T3 trains 16 shared linear-head parameters from twelve deterministic one-hot context vectors, of which six masked positions cover all four labels. Adam performs 100 updates at `.08`. PyTorch 2.14.1 CPU float64 measured CE `1.38629436 → .01292025` and accuracy `1.0`; acceptance is CE below `.03` and below `.03` of baseline, with exact training-label accuracy. The common seed is 42 and native thread cap is two; this experiment uses no random sampling. These context features are deliberately easy to classify and demonstrate optimization of the submitted function, not generalization or tokenizer quality. Numerical gradient checks separately prevent a training-only shortcut from satisfying the module.
