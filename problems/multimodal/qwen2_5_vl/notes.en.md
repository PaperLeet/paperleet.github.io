# Qwen2.5-VL visual window permutation — Solution

View the original grid as merge_size-square cells, then group these cells into windows. Preserve row-major patch order inside each cell and count cumulative lengths in original patches. Boundary windows contain only real cells, with no padding indices. The experiment gathers window means, scatters them back, and compares against independent spatial slices.

$$
\operatorname{index}(t,h,w)=(tH+h)W+w,\quad c_{j+1}=c_j+\#\{\text{valid original patches in window }j\}.
$$

The reference implementation follows the declared layout and reduction:

```python
def window_permutation(frames: int, height: int, width: int, merge_size: int, window_size: int) -> tuple[Tensor, Tensor]:
    args=(frames,height,width,merge_size,window_size)
    if any(type(v)!=int or v<=0 for v in args) or height%merge_size or width%merge_size:
        raise ValueError("positive integers and grid divisible by merge_size required")
    mh,mw=height//merge_size,width//merge_size
    order=[];bounds=[0]
    for frame in range(frames):
        for wh in range(0,mh,window_size):
            for ww in range(0,mw,window_size):
                for cell_h in range(wh,min(wh+window_size,mh)):
                    for cell_w in range(ww,min(ww+window_size,mw)):
                        for dy in range(merge_size):
                            for dx in range(merge_size):
                                order.append(((frame*height+cell_h*merge_size+dy)*width)+cell_w*merge_size+dx)
                bounds.append(len(order))
    return torch.tensor(order,dtype=torch.long),torch.tensor(bounds,dtype=torch.long)
```

Common errors caught by mutation tests:

- Reuses first-frame indices for every frame
- Counts merged cells rather than original patches

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Window grouping, spatial-merge ordering, and cumulative lengths for dynamic resolutions; exclude the complete ViT and GPU kernels.

- [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
