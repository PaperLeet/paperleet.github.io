import math
import torch
import torch.nn.functional as F


def measure(fn):
    q=torch.zeros(1,2,dtype=torch.float64);v=torch.eye(2,dtype=torch.float64);mask=torch.tensor([True]);instruction=torch.zeros(1,2,dtype=torch.float64,requires_grad=True);target=q.new_tensor([[1.5,.2]]);curve=[]
    for _ in range(81):
        out=fn(q,instruction,v,mask);loss=(out-target).square().mean();curve.append(loss.item());g,=torch.autograd.grad(loss,instruction);instruction=(instruction-.5*g).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"feature_margin":float(out[0,0]-out[0,1])}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'InstructBLIP instruction-aware query block', 'problem': 'multimodal/instructblip', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2305.06500', 'anchor': 'Section 3.2 and Figure 2: instruction-aware visual feature extraction with query-text self-attention', 'adaptation': 'Two-stage query self-attention and visual cross-attention with identity projections, testing instruction influence and padding isolation; not a full Q-Former reproduction. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "InstructBLIP instruction-aware query block", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
