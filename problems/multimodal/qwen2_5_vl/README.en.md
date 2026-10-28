# Qwen2.5-VL visual window permutation

## 1. Problem

Implement `window_permutation(frames: int, height: int, width: int, merge_size: int, window_size: int) -> tuple[Tensor, Tensor]`.

Window grouping, spatial-merge ordering, and cumulative lengths for dynamic resolutions; exclude the complete ViT and GPU kernels.

## 2. Core equations

$$
\operatorname{index}(t,h,w)=(tH+h)W+w,\quad c_{j+1}=c_j+\#\{\text{valid original patches in window }j\}.
$$

## 3. From equations to code

View the original grid as merge_size-square cells, then group these cells into windows. Preserve row-major patch order inside each cell and count cumulative lengths in original patches. Boundary windows contain only real cells, with no padding indices. The experiment gathers window means, scatters them back, and compares against independent spatial slices.

## 4. Conventions

All arguments are positive integers. height/width describe the original patch grid and are divisible by merge_size; window_size counts merged cells. Iterate frames, window rows/columns, merged-cell rows/columns within a window, then patch rows/columns within a cell. Return original row-major patch indices without boundary padding. Outputs are CPU int64 permutation [T*H*W] and cu_seqlens counted in original patches, starting at zero.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/qwen2_5_vl --impl problems/multimodal/qwen2_5_vl/solution.py
paperleet mutate multimodal/qwen2_5_vl
paperleet vizdata multimodal/qwen2_5_vl
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.1.1 Fast and Efficient Vision Encoder: window attention and spatial merging

- [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
