# Qwen2.5-Omni time-aligned audio-video interleaving

## 1. Problem

Implement `time_aligned_interleave(video_times: Tensor, audio_times: Tensor) -> tuple[Tensor, Tensor]`.

Video-first stream interleaving within two-second chunks and a shared 40 ms temporal grid; exclude audiovisual encoders, Thinker–Talker, and waveform generation.

## 2. Core equations

$$
c(t)=\lfloor t/2\rfloor,\quad p_t=\lfloor25t\rfloor,\quad \text{order key}=(c(t),\ \text{video before audio},\ \text{stream index}).
$$

## 3. From equations to code

Assign real timestamps to half-open two-second chunks and sort by chunk, modality, then original index. Video-first order is not global timestamp order: audio retains its own absolute time IDs despite coming later within a chunk. Forty milliseconds means 25 positions per second; video frame numbers are not timestamps. Quantized metadata is nondifferentiable, while downstream embedding gathers still propagate feature gradients.

## 4. Conventions

video_times [V] and audio_times [A] are separately nondecreasing, nonnegative floating seconds, with at least one nonempty stream and shared dtype/device. Chunks are half-open [2c,2c+2). Order chunks increasingly; place all video before all audio within each chunk, preserving each stream’s order. Return int64 permutation into concat(video,audio) and reordered temporal_ids=floor(time*25). Spatial H/W axes are excluded; quantized times and indices are not differentiable.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/qwen2_5_omni --impl problems/multimodal/qwen2_5_omni/solution.py
paperleet mutate multimodal/qwen2_5_omni
paperleet vizdata multimodal/qwen2_5_omni
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 2.2 Video and TMRoPE: 40 ms temporal IDs and video-first 2-second chunks

- [https://arxiv.org/abs/2503.20215](https://arxiv.org/abs/2503.20215)
