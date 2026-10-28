import math
import torch
import torch.nn.functional as F


def measure(fn):
    errors=[];windows=0
    for frames,height,width in ((1,4,6),(2,6,8),(1,8,10)):
        permutation,bounds=fn(frames,height,width,2,2);x=torch.arange(frames*height*width,dtype=torch.float64);restored=torch.zeros_like(x)
        for a,b in zip(bounds.tolist(),bounds.tolist()[1:]):
            ids=permutation[a:b];restored[ids]=x[ids].mean();windows+=1
        oracle=[]
        for t in range(frames):
            for h in range(height):
                for w in range(width):
                    top,left=h//4*4,w//4*4;values=[float((t*height+hh)*width+ww) for hh in range(top,min(top+4,height)) for ww in range(left,min(left+4,width))]
                    oracle.append(sum(values)/len(values))
        errors.append(float((restored-x.new_tensor(oracle)).abs().max()))
    return errors,{"max_window_mean_error":max(errors),"windows":windows}


def run(fn):
    curve, metrics = measure(fn)
    return {'kind': 'toy_experiment', 'title': 'Qwen2.5-VL visual window permutation', 'problem': 'multimodal/qwen2_5_vl', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2502.13923', 'anchor': 'Section 2.1.1 Fast and Efficient Vision Encoder: window attention and spatial merging', 'adaptation': 'Window grouping, spatial-merge ordering, and cumulative lengths for dynamic resolutions; exclude the complete ViT and GPU kernels. Deterministic CPU float64 synthetic inputs; no benchmark reproduction.'}} | {"metrics": metrics, "charts": [{"kind": "line", "title": "Qwen2.5-VL visual window permutation", "x_label": "Step / deterministic case", "x": list(range(len(curve))), "series": [{"name": "Measured signal", "values": curve}]}]}
