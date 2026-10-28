import math
import torch
import torch.nn.functional as F


def measure(fn):
    raw=torch.eye(2,dtype=torch.float64);w=torch.zeros(2,2,dtype=torch.float64,requires_grad=True);text=torch.zeros(3,2,dtype=torch.float64);labels=torch.tensor([-100,-100,7]);target=torch.tensor([1.,-1.],dtype=torch.float64);curve=[]
    for _ in range(61):
        packed,lab=fn(text,raw@w,labels,1);prediction=packed[1:3].mean(0);loss=(prediction-target).square().mean();curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.3*g).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"supervised_tokens":int((lab!=-100).sum())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'LLaVA visual-token insertion and label alignment', 'problem': 'multimodal/llava', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2304.08485', 'anchor': 'Section 4.1 architecture; official llava_arch.py prepare_inputs_labels_for_multimodal', 'adaptation': 'Pack one image into one sample and align supervision labels; features are already projected, excluding the vision encoder, language model, and data construction. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "LLaVA visual-token insertion and label alignment", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
