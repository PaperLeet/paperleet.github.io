# Qwen2-VL chunked MRoPE

## 1. Problem

Implement `chunked_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor`.

Contiguous frequency chunks for three-axis rotary positions in an equal-axis teaching configuration; exclude the dynamic-resolution encoder.

## 2. Core equations

$$
a(r)=\lfloor r/(R/3)\rfloor,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

## 3. From equations to code

Split the R frequencies into three contiguous ranges. Keep the original frequency index r and select only its axis-specific position; do not restart frequencies for each axis. Each orthogonal rotation pairs coordinates r and r+R, preserving norm and inverting under negative positions. Equal axis widths isolate the difference from interleaved allocation.

## 4. Conventions

x is [N,2R], positions [3,N] ordered time/height/width, and inv_freq [R]. R is a positive multiple of 3. This equal-width teaching configuration assigns the first/middle/last R/3 frequencies to time/height/width. Rotate matching coordinates in the first and second feature halves, not adjacent coordinates. Floating positions are differentiable; finite inputs share dtype/device.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/qwen2_vl --impl problems/multimodal/qwen2_vl/solution.py
paperleet mutate multimodal/qwen2_vl
paperleet vizdata multimodal/qwen2_vl
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.1: Multimodal Rotary Position Embedding

- [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
