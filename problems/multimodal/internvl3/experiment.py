import math
import torch
import torch.nn.functional as F


def measure(fn):
    visual=torch.tensor([False]+[True]*64+[False]);ids=torch.tensor([-1]+[0]*64+[-1]);curve=[]
    for delta in (1.,.5,.25,.125,.0625):
        positions=fn(visual,ids,torch.tensor([delta],dtype=torch.float64));curve.append(float(positions[-1]))
    return curve,{"full_span":curve[0],"compressed_span":curve[-1],"text_gap":float(positions[-1]-positions[-2]),"visual_gap":float(positions[2]-positions[1])}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'InternVL3 variable visual position increments', 'problem': 'multimodal/internvl3', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2504.10479', 'anchor': 'Section 2.1: Variable Visual Position Encoding, Equations 3 and 4', 'adaptation': 'Per-image position increments, cumulative multimodal offsets, and gradients with an explicit origin of -1; exclude full rotary embeddings and InternVL training. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "InternVL3 variable visual position increments", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
