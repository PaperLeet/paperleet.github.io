# Qwen2.5-Omni 音视频时间对齐与交错 — 题解

先用实际秒数确定两秒半开区间，再按块、模态、原索引排序。块内视频优先不是全局时间排序；音频 token 虽然排在视频后面，仍保留自己的绝对时间 ID。40ms 对应每秒 25 个位置，不能把视频帧号当成时间。索引量化本身无导数，下游 embedding gather 仍可正常传播特征梯度。

$$
c(t)=\lfloor t/2\rfloor,\quad p_t=\lfloor25t\rfloor,\quad \text{order key}=(c(t),\ \text{video before audio},\ \text{stream index}).
$$

参考实现按上述布局和归约逐步计算：

```python
def time_aligned_interleave(video_times: Tensor, audio_times: Tensor) -> tuple[Tensor, Tensor]:
    _floating(video_times,audio_times)
    if video_times.ndim!=1 or audio_times.ndim!=1 or video_times.numel()+audio_times.numel()==0 or (video_times<0).any() or (audio_times<0).any():raise ValueError("expected nonnegative timestamp vectors with at least one token")
    for times in (video_times,audio_times):
        if times.numel()>1 and (times[1:]<times[:-1]).any():raise ValueError("timestamps must be sorted")
    video=video_times.tolist();audio=audio_times.tolist();entries=[]
    for i,t in enumerate(video):entries.append((math.floor(t/2),0,i,i,t))
    for i,t in enumerate(audio):entries.append((math.floor(t/2),1,i,len(video)+i,t))
    entries.sort(key=lambda row:(row[0],row[1],row[2]))
    order=torch.tensor([row[3] for row in entries],dtype=torch.long,device=video_times.device)
    temporal=torch.tensor([math.floor(row[4]*25) for row in entries],dtype=torch.long,device=video_times.device)
    return order,temporal
```

错误实现检测覆盖这些常见问题：

- Uses 20 ms instead of 40 ms time IDs
- Places audio before video within a chunk

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

两秒块内视频优先的流交错和 40ms 时间格映射；不包含音视频编码器、Thinker–Talker 或语音波形生成。

- [https://arxiv.org/abs/2503.20215](https://arxiv.org/abs/2503.20215)
