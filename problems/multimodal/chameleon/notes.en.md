# Chameleon per-head query-key LayerNorm — Solution

Compute statistics independently for each token and head, never across tokens or heads. RMSNorm cannot replace mean-centering here. Use population variance with denominator D, divide by sqrt(var+eps), then apply per-head affine parameters. A constant head therefore outputs bias. Normalizing growing query/key norms keeps attention from rapidly saturating.

$$
\mu_{th}=D^{-1}\sum_d x_{thd},\quad \sigma^2_{th}=D^{-1}\sum_d(x_{thd}-\mu_{th})^2,\quad y_{thd}=\gamma_{hd}\frac{x_{thd}-\mu_{th}}{\sqrt{\sigma^2_{th}+\epsilon}}+\beta_{hd}.
$$

The reference implementation follows the declared layout and reduction:

```python
def qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    _floating(x,weight,bias)
    if x.ndim!=3 or x.numel()==0 or weight.shape!=x.shape[1:] or bias.shape!=weight.shape or eps<=0:
        raise ValueError("expected (T,H,D), (H,D) affine parameters, positive eps")
    centered=x-x.mean(dim=-1,keepdim=True)
    variance=centered.square().mean(dim=-1,keepdim=True)
    return centered/torch.sqrt(variance+eps)*weight+bias
```

Common errors caught by mutation tests:

- Omits mean centering
- Uses sample variance instead of population variance

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Per-head LayerNorm statistics, population variance, and affine parameters, with an attention-entropy experiment under growing input norms.

- [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py)
