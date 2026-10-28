import math
import torch
import torch.nn.functional as F


def measure(fn):
    x=torch.zeros(3,5,dtype=torch.float64,requires_grad=True);p=torch.tensor([1,4,0]);teacher=x.new_tensor([[0.,5.,0.,0.,0.],[0.,0.,0.,0.,5.],[5.,0.,0.,0.,0.]])
    curve=[]
    for _ in range(81):
        loss=fn(x,teacher,p,.3);curve.append(loss.item());g,=torch.autograd.grad(loss,x);x=(x-.7*g).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"accuracy":float((x.argmax(-1)==p).double().mean())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'BLIP momentum soft-target mixing', 'problem': 'multimodal/blip', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2201.12086', 'anchor': 'Section 3.1: image-text contrastive learning with momentum encoders and soft targets', 'adaptation': 'Implement BLIP momentum soft-target distillation for one retrieval direction with rectangular negative queues; omit queue updates, momentum encoders, and CapFilt data generation. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "BLIP momentum soft-target mixing", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
