# Gradient Accumulation: Valid-token Weighting

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Nonempty equal-length lists. Each gradient is a finite float32/64 CPU tensor for the same parameter, with identical shape/dtype/device, possibly scalar or empty. Each token_count is a nonnegative integer and their total is >0. Input gradient i already represents the mean over its valid tokens. Return sum(count_i * gradient_i)/sum(count_i), ignoring zero-count batches, same shape/dtype/device. Do not average microbatches equally, divide by the number of accumulation steps, perform optimizer updates, mutate inputs or retain state. Higher-order gradients are not required. Gradients from different parameters are not being merged.

```python
accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor
```

$$
g=\frac{\sum_i n_i g_i}{\sum_i n_i}
$$

## Example

```python
accumulate_gradients([torch.tensor([1., 3.]), torch.tensor([5., 7.])], [1, 3])
# tensor([4., 6.])
```

## Before implementing

Each input is already a local token mean: weight it by its token count and divide once by the global count. Empty batches do not dilute the denominator. The experiment compares one update from unequal microbatches with a full-batch update.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
