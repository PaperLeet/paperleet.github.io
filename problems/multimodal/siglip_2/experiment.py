import math
import torch
import torch.nn.functional as F


def measure(fn):
    z=torch.zeros(1,4,3,dtype=torch.float64,requires_grad=True)
    q=z.new_tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]]);mask=torch.tensor([[True,False,True,False]])
    curve=[]
    for _ in range(61):
        loss=fn(z,q,mask,.7);curve.append(loss.item())
        grad,=torch.autograd.grad(loss,z);z=(z-.3*grad).detach().requires_grad_()
    return curve,{"initial":curve[0],"final":curve[-1],"unmasked_movement":(z[0,1].abs().sum()+z[0,3].abs().sum()).detach().item()}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'SigLIP 2 masked patch self-distillation', 'problem': 'multimodal/siglip_2', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2502.14786', 'anchor': 'Section 2.3: masked prediction uses stopped teacher targets at masked patch locations', 'adaptation': 'Implement the probability-distillation kernel of SigLIP 2 masked prediction with supplied teacher targets; exclude EMA encoders, augmentation, projection heads, and the full training recipe. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "SigLIP 2 masked patch self-distillation", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
