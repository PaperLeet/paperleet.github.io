import math
import torch
import torch.nn.functional as F


def measure(fn):
    x=torch.eye(4,dtype=torch.float64)
    y=(torch.eye(4,dtype=torch.float64).flip(0)+.1).requires_grad_()
    curve=[]
    for _ in range(61):
        loss=fn(x,y,x.new_tensor(.3));curve.append(loss.item())
        grad,=torch.autograd.grad(loss,y);y=(y-.4*grad).detach().requires_grad_()
    return curve, {"initial":curve[0],"final":curve[-1],"retrieval_accuracy":float((y.argmax(-1)==torch.arange(4)).double().mean())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'ALIGN bidirectional temperature contrast', 'problem': 'multimodal/align', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2102.05918', 'anchor': 'Section 3: normalized embeddings and symmetric image-text contrastive loss', 'adaptation': 'Normalized dual-encoder features, differentiable temperature, and bidirectional batch softmax; train synthetic image-text pairs without web-scale noisy-data collection. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "ALIGN bidirectional temperature contrast", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
