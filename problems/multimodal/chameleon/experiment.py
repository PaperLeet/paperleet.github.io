import math
import torch
import torch.nn.functional as F


def measure(fn):
    base=torch.tensor([[[1.,.2,-.7,.4]],[[.2,1.,.4,-.3]],[[.4,-.2,1.,.8]]],dtype=torch.float64);w=torch.ones(1,4,dtype=torch.float64);b=w*0;curve=[];raw=[]
    for scale in (1.,2.,5.,10.,50.,100.):
        normalized=fn(base*scale,w,b,1e-5)[:,0];p=torch.softmax(normalized@normalized.T/2,-1);curve.append(float(-(p*torch.log(p)).sum(-1).mean()))
        raw_p=torch.softmax((base[:,0]*scale)@(base[:,0]*scale).T/2,-1);raw.append(float(-(raw_p*torch.log(raw_p.clamp_min(1e-30))).sum(-1).mean()))
    return curve,{"entropy_spread":max(curve)-min(curve),"normalized_entropy":curve[-1],"raw_entropy":raw[-1]}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'Chameleon per-head query-key LayerNorm', 'problem': 'multimodal/chameleon', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2405.09818', 'anchor': 'Section 2.3 Stability; ChameleonLayerNorm computes statistics only over head_dim', 'adaptation': 'Per-head LayerNorm statistics, population variance, and affine parameters, with an attention-entropy experiment under growing input norms. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "Chameleon per-head query-key LayerNorm", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
