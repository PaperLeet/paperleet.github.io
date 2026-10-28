import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

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
