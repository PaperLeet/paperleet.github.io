# Top-k: Temperature and Candidate Probabilities

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Finite float32/64 logits [...,V], V>=1, CPU; zero-length leading dimensions allowed. Integer 1<=k<=V; finite temperature>0. Select exactly k largest logits per last-axis row, breaking ties by smaller vocabulary index. Return same-shaped/dtype/device probabilities: softmax(logits/temperature) restricted to selected indices, exactly zero elsewhere, row sums one. Selection is independent for each row. No random draw or input mutation; derivatives are not required. Exact-k tie handling is an explicit exercise convention, unlike threshold-based implementations that may retain extra ties.

Compute scaling in float64 and cast only the final probabilities to the input dtype.

```python
top_k_probs(logits: Tensor, k: int, temperature: float = 1.) -> Tensor
```

$$
q_i=\frac{\mathbf{1}[i\in K]\exp((z_i-\max z)/\tau)}{\sum_{j\in K}\exp((z_j-\max z)/\tau)}
$$

## Example

```python
top_k_probs(torch.tensor([2., 2., 1.]), 1)
# tensor([1., 0., 0.])
```

## Before implementing

Select using a deterministic tie order, mask other logits to negative infinity, then apply stable softmax. Do not divide probabilities by temperature. For temperature >=1, divide logits before centering to avoid overflowing opposite-sign differences; for temperature <1, center before division. Use float64 intermediates and cast the final result back.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

## Numerical range and measured sampling

Use float64 intermediates so a positive Python temperature is not rounded to zero in float32. At temperature >=1 divide before subtracting the maximum; at temperature <1 subtract before dividing. Negative overflow then represents a vanishing weight, while the maximum stays zero. Cast the final probabilities back to the input dtype.

T3 samples 8192 token IDs from your returned distribution with a fixed seed, checks that excluded tokens never appear, and plots measured frequencies against independently specified probabilities. The statistical tolerance is for sampled frequencies; deterministic probability checks remain strict. This is a small synthetic experiment, without a trained language model or text-quality benchmark.
