import math
import torch
import torch.nn.functional as F


def measure(fn):
    coordinates=torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.],[1.,1.,1.],[2.,1.,0.]],dtype=torch.float64).T
    base=torch.cat([torch.ones(6,9,dtype=torch.float64),torch.zeros(6,9,dtype=torch.float64)],-1);freq=torch.tensor([1.,.9,.8,.6,.5,.4,.3,.2,.1],dtype=torch.float64)
    encoded=fn(base,coordinates,freq);query_ids=torch.tensor([4,1,5,0,3,2]);queries=fn(base,coordinates[:,query_ids],freq);scores=queries@encoded.T
    curve=[float(scores[i,query_ids[i]]) for i in range(6)]
    return curve,{"retrieval_accuracy":float((scores.argmax(-1)==query_ids).double().mean()),"norm_error":float((encoded.square().sum(-1)-base.square().sum(-1)).abs().max())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'Qwen2-VL chunked MRoPE', 'problem': 'multimodal/qwen2_vl', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2409.12191', 'anchor': 'Section 2.1: Multimodal Rotary Position Embedding', 'adaptation': 'Contiguous frequency chunks for three-axis rotary positions in an equal-axis teaching configuration; exclude the dynamic-resolution encoder. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "Qwen2-VL chunked MRoPE", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
