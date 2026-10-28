# Flamingo media-causal gated attention — Solution

Media IDs identify the immediately preceding medium, requiring equality rather than a less-than relation. Avoid softmax on all-negative-infinity rows by replacing them with safe rows and masking the resulting weights to zero. A tanh gate starts at an identity text path while retaining a nonzero derivative that can learn visual contributions.

$$
y_i=x_i+\tanh(g)\sum_{j:m_j=t_i>0}\operatorname{softmax}_{j}(x_i^Tk_j/\sqrt D)v_j.
$$

The reference implementation follows the declared layout and reduction:

```python
def gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor:
    _floating(x, keys, values, gate)
    if x.ndim!=2 or keys.ndim!=2 or x.numel()==0 or keys.numel()==0 or values.shape!=keys.shape or x.shape[1]!=keys.shape[1] or text_media.shape!=(x.shape[0],) or visual_media.shape!=(keys.shape[0],) or gate.ndim!=0:
        raise ValueError("incompatible attention shapes")
    if text_media.dtype!=torch.long or visual_media.dtype!=torch.long or (text_media<0).any() or (visual_media<=0).any(): raise ValueError("invalid media IDs")
    mask=(text_media[:,None]==visual_media[None,:]) & (text_media[:,None]>0)
    scores=(x@keys.T)/math.sqrt(x.shape[-1])
    masked=scores.masked_fill(~mask,float('-inf'))
    safe=torch.where(mask.any(-1,keepdim=True),masked,torch.zeros_like(scores))
    weights=torch.softmax(safe,-1)*mask.to(x.dtype)
    return x+torch.tanh(gate)*(weights@values)
```

Common errors caught by mutation tests:

- Attends all earlier media
- Omits tanh bound

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Immediately preceding media masks, safe softmax for rows without media, and zero-initialized tanh gating in single-head identity-projection cross-attention.

- [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
