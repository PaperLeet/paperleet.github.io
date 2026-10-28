import math
import torch
import torch.nn.functional as F


def measure(fn):
    x=torch.zeros(3,2,dtype=torch.float64);keys=torch.ones(2,2,dtype=torch.float64);values=x.new_tensor([[1.,2.],[-2.,1.]]);tm=torch.tensor([0,1,2]);vm=torch.tensor([1,2]);gate=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    target=torch.cat([x[:1],values*math.tanh(1.)],0);curve=[]
    for _ in range(61):
        out=fn(x,keys,values,tm,vm,gate);loss=(out-target).square().mean();curve.append(loss.item());g,=torch.autograd.grad(loss,gate);gate=(gate-.2*g).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"prefix_change":float(out[0].abs().sum())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'Flamingo media-causal gated attention', 'problem': 'multimodal/flamingo', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2204.14198', 'anchor': 'Section 2.2 and Appendix A: single-image causal masking and tanh-gated cross-attention', 'adaptation': 'Immediately preceding media masks, safe softmax for rows without media, and zero-initialized tanh gating in single-head identity-projection cross-attention. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "Flamingo media-causal gated attention", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
