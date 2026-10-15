# Global Gradient Clipping: One Norm for All Parameters

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Nonempty list of finite float32/64 CPU gradient tensors, same dtype/device but arbitrary shapes, including scalars/empty tensors. Finite max_norm>=0 and eps>0. Compute the global L2 norm over every element of every tensor. Return (new gradient list, original scalar total_norm), same shapes/dtype/device, using scale=min(1,max_norm/(total_norm+eps)) for every tensor. Empty/all-zero gradients have norm zero; max_norm=0 yields zeros. Pure function: no .grad mutation, optimizer update or input/list mutation. No higher-order-gradient requirement. The denominator epsilon is explicit and matches the exercise clipping convention; this is norm clipping, not elementwise value clipping.

```python
clip_gradients(gradients: list[Tensor], max_norm: float, eps: float = 1e-6) -> tuple[list[Tensor], Tensor]
```

$$
N=\sqrt{\sum_i\|g_i\|_2^2},\quad s=\min(1, M/(N+\epsilon)),\quad \tilde g_i=s g_i
$$

## Example

```python
clip_gradients([torch.tensor([3.]), torch.tensor([4.])], 2.5)
# ([tensor([1.4999997]), tensor([1.9999996])], tensor(5.))
```

## Before implementing

Sum squared entries across every parameter, then compute one norm and one scale. Return the original norm, not the clipped norm. Never amplify a gradient that is below the limit.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)
