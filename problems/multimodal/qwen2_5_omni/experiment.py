import math
import torch
import torch.nn.functional as F


def measure(fn):
    video=torch.tensor([0.,.12,.4,2.,2.4,4.],dtype=torch.float64);audio=torch.tensor([0.,.04,.12,.4,2.,2.04,2.4,4.],dtype=torch.float64)
    order,temporal=fn(video,audio);restored=torch.empty_like(temporal);restored[order]=temporal
    matched=[(0,0),(1,2),(2,3),(3,4),(4,6),(5,7)];errors=[abs(int(restored[i])-int(restored[len(video)+j])) for i,j in matched]
    keys=[]
    for index in order.tolist():
        t=float(video[index]) if index<len(video) else float(audio[index-len(video)]);keys.append((math.floor(t/2),0 if index<len(video) else 1))
    return [float(v) for v in temporal.tolist()],{"alignment_error":max(errors),"chunk_order_valid":keys==sorted(keys),"last_time_id":int(temporal[-1])}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'Qwen2.5-Omni time-aligned audio-video interleaving', 'problem': 'multimodal/qwen2_5_omni', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2503.20215', 'anchor': 'Section 2.2 Video and TMRoPE: 40 ms temporal IDs and video-first 2-second chunks', 'adaptation': 'Video-first stream interleaving within two-second chunks and a shared 40 ms temporal grid; exclude audiovisual encoders, Thinker–Talker, and waveform generation. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "Qwen2.5-Omni time-aligned audio-video interleaving", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
