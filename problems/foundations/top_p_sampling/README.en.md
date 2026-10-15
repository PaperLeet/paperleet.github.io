# Top-p: Cumulative Probability and Nucleus Candidates

A standalone LLM exercise. Supplied code and synthetic experiments do not require any previously solved exercise.

## Task and conventions

Finite float32/64 logits [...,V], V>=1, CPU; empty leading dimensions allowed. Finite 0<p<=1 and temperature>0. Form full-vocabulary softmax(logits/temperature), sort descending with smaller vocabulary index first on ties, and retain the shortest nonempty prefix whose cumulative mass is >=p. Include the token that first reaches/crosses p. At p=1 keep all tokens. Renormalize retained probabilities and return original vocabulary order with zeros elsewhere, same shape/dtype/device. No random sampling, input mutation or derivative requirement. Compute scaling, probabilities and cumulative boundary comparisons in float64, without arbitrary tolerances; cast only the final result to the input dtype.

```python
top_p_probs(logits: Tensor, p: float, temperature: float = 1.) -> Tensor
```

$$
K=\min\{k:\sum_{i=1}^k p_{(i)}\ge p\},\quad q_{(i)}=\frac{p_{(i)}\mathbf{1}[i\le K]}{\sum_{j=1}^Kp_{(j)}}
$$

## Example

```python
top_p_probs(torch.zeros(4), .5)
# tensor([.5, .5, 0., 0.])
```

## Before implementing

Compare the cumulative mass before each token to p. Keeping only cumulative sums <=p drops the crossing token. Apply temperature before selecting the nucleus and restore vocabulary order afterward.

## Tests and experiment

T0 checks the interface; T1 compares an independent oracle; T2 checks boundaries, invariants, applicable gradients and input preservation; T3 runs your function in the supplied experiment. Charts show measured outputs, not full-model training or speed claims.

## References

- [Primary documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

## Numerical range and measured sampling

Use float64 intermediates so a positive Python temperature is not rounded to zero in float32. At temperature >=1 divide before subtracting the maximum; at temperature <1 subtract before dividing. Negative overflow then represents a vanishing weight, while the maximum stays zero. Cast the final probabilities back to the input dtype.

T3 samples 8192 token IDs from your returned distribution with a fixed seed, checks that excluded tokens never appear, and plots measured frequencies against independently specified probabilities. The statistical tolerance is for sampled frequencies; deterministic probability checks remain strict. This is a small synthetic experiment, without a trained language model or text-quality benchmark.

## Paper definition and exact boundaries

This exercise follows the shortest-prefix definition in [Holtzman et al., ICLR 2020, §3.1, Eq. (2–3)](https://arxiv.org/abs/1904.09751): cumulative mass >=p. The [authors’ released filtering code](https://github.com/ari-holtzman/degen/blob/0acfd2d0ba8484e24e9c5241f75f34be15ef2609/gen.py#L118-L132) shifts a strict >p mask and retains one additional token at exact equality. Here uniform four-token probabilities with p=.5 retain exactly two. Probability ordering and boundary comparisons use float64; exact ties keep smaller vocabulary indices. This is an explicit teaching convention, not a literal reproduction of the released filter.
