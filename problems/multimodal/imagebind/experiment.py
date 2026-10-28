import math
import torch
import torch.nn.functional as F


def measure(fn):
    image=torch.eye(3,dtype=torch.float64);modalities=torch.stack([image.flip(0)+.2,image.roll(1,0)+.1]).requires_grad_();curve=[]
    for _ in range(101):
        loss=fn(image,modalities,image.new_tensor(.25));curve.append(loss.item());g,=torch.autograd.grad(loss,modalities);modalities=(modalities-.5*g).detach().requires_grad_()
    cross=F.normalize(modalities[0],dim=-1)@F.normalize(modalities[1],dim=-1).T
    return curve,{"initial":curve[0],"final":curve[-1],"cross_modal_accuracy":float((cross.argmax(-1)==torch.arange(3)).double().mean())}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'ImageBind image-anchored multimodal objective', 'problem': 'multimodal/imagebind', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2305.05665', 'anchor': 'Section 3 and Equation 1: binding modalities using image-paired contrastive learning', 'adaptation': 'Implement the image-anchored pairing graph and reduction across modalities; align synthetic modalities without direct nonimage pairs, excluding the six encoders. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "ImageBind image-anchored multimodal objective", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
