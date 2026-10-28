# InternVL3 variable visual position increments — Solution

The recurrence depends on the current token’s modality and equals a prefix sum of increments minus one. Use one delta per image and never reset at image or text boundaries. Delta gradients count how many subsequent prefixes include each visual token. Delta one recovers positions 0 through L-1; a first visual token starts at delta-1 under the explicit origin.

$$
p_i=-1+\sum_{j=0}^{i}\left[(1-v_j)+v_j\delta_{\operatorname{image}(j)}\right].
$$

The reference implementation follows the declared layout and reduction:

```python
def variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor:
    _floating(deltas)
    if visual.ndim!=1 or visual.dtype!=torch.bool or image_ids.shape!=visual.shape or image_ids.dtype!=torch.long or deltas.ndim!=1 or (deltas<=0).any() or (deltas>1).any():
        raise ValueError("invalid masks, image IDs or deltas")
    if (image_ids[~visual]!=-1).any() or (image_ids[visual]<0).any() or (image_ids[visual]>=deltas.numel()).any():raise ValueError("invalid image assignment")
    increments=[]
    for is_visual,index in zip(visual.tolist(),image_ids.tolist()):
        increments.append(deltas[index] if is_visual else deltas.new_tensor(1.))
    if not increments:return deltas.new_empty((0,))
    return torch.stack(increments).cumsum(0)-1
```

Common errors caught by mutation tests:

- Uses ordinary integer positions for images
- Starts positions one unit too late

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Per-image position increments, cumulative multimodal offsets, and gradients with an explicit origin of -1; exclude full rotary embeddings and InternVL training.

- [https://arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479)
