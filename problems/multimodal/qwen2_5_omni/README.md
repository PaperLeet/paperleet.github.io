# Qwen2.5-Omni 音视频时间对齐与交错

## 1. 问题

实现 `time_aligned_interleave(video_times: Tensor, audio_times: Tensor) -> tuple[Tensor, Tensor]`。

两秒块内视频优先的流交错和 40ms 时间格映射；不包含音视频编码器、Thinker–Talker 或语音波形生成。

## 2. 核心公式

$$
c(t)=\lfloor t/2\rfloor,\quad p_t=\lfloor25t\rfloor,\quad \text{order key}=(c(t),\ \text{video before audio},\ \text{stream index}).
$$

## 3. 公式到代码

先用实际秒数确定两秒半开区间，再按块、模态、原索引排序。块内视频优先不是全局时间排序；音频 token 虽然排在视频后面，仍保留自己的绝对时间 ID。40ms 对应每秒 25 个位置，不能把视频帧号当成时间。索引量化本身无导数，下游 embedding gather 仍可正常传播特征梯度。

## 4. 本题约定

video_times=[V]、audio_times=[A] 是各自非降序的非负秒数浮点张量，至少一侧非空；输入共享 dtype/device。把区间 [2c,2c+2) 视作 chunk，按 chunk 升序，chunk 内所有视频先于所有音频，各模态内部保持输入顺序。返回 int64 permutation，对应 concat(video,audio) 的原索引，以及重排后的 temporal_ids=floor(time*25)。不含空间 H/W 轴；量化时间与索引不反传。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/qwen2_5_omni --impl problems/multimodal/qwen2_5_omni/solution.py
paperleet mutate multimodal/qwen2_5_omni
paperleet vizdata multimodal/qwen2_5_omni
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.2 Video and TMRoPE: 40 ms temporal IDs and video-first 2-second chunks

- [https://arxiv.org/abs/2503.20215](https://arxiv.org/abs/2503.20215)
