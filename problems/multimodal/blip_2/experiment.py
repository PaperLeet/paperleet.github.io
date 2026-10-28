import math
import torch
import torch.nn.functional as F


def measure(fn):
    q=torch.tensor([[[1.,.4],[.3,1.]],[[.2,-1.],[-1.,.3]]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[1.,0.],[-1.,0.]],dtype=torch.float64);curve=[]
    for _ in range(61):
        scores=fn(q,t,q.new_tensor(.2));loss=-F.log_softmax(scores,-1).diag().mean();curve.append(loss.item())
        g,=torch.autograd.grad(loss,q);q=(q-.5*g).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"accuracy":float((scores.argmax(-1)==torch.arange(2)).double().mean())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'BLIP-2 maximum-query image-text matching', 'problem': 'multimodal/blip_2', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2301.12597', 'anchor': 'Section 3.1: image-text contrastive learning takes the maximum similarity over query outputs', 'adaptation': 'Normalize Q-Former outputs, compute per-text maximum-query similarities, and route gradients through winning queries; omit Transformer and frozen large-model execution. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "BLIP-2 maximum-query image-text matching", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
