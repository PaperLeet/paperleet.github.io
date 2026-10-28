# Qwen3-VL interleaved MRoPE

## 1. Problem

Implement `interleaved_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor`.

Interleaved frequency-axis allocation and half-split rotary pairs in an equal-axis teaching configuration; exclude DeepStack and the full video model.

## 2. Core equations

$$
a(r)=r\bmod3,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

## 3. From equations to code

Cycle time, height, and width across frequency indices so each axis spans multiple frequency bands. Multiply each selected position by its matching frequency and use half-split rotations. Different axis positions make contiguous allocation observably wrong; one-dimensional RoPE tests would miss this, so the oracle uses distinct coordinates per axis.

## 4. Conventions

x is [N,2R], positions [3,N] ordered time/height/width, and inv_freq [R]. R is a positive multiple of 3. This equal-width teaching configuration assigns frequency r to axis r%3. Rotate matching coordinates in the first and second feature halves, not adjacent coordinates. Floating positions are differentiable; finite inputs share dtype/device.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/qwen3_vl --impl problems/multimodal/qwen3_vl/solution.py
paperleet mutate multimodal/qwen3_vl
paperleet vizdata multimodal/qwen3_vl
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.1 Interleaved MRoPE

- [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
