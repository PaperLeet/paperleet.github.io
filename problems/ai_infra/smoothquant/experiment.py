import torch

def quantize(t):
    step=t.abs().max().clamp(min=1e-12)/127
    return torch.round(t/step).clamp(-127,127)*step

def evaluate(smooth_scales,smooth_transform):
    x=torch.sin(torch.arange(68,dtype=torch.float64).reshape(17,4)*.37)*torch.tensor([100.,1.,1.,1.],dtype=torch.float64)
    w=torch.cos(torch.arange(12,dtype=torch.float64).reshape(4,3)*.41)*torch.tensor([.01,1.,1.,1.],dtype=torch.float64)[:,None]
    target=x@w; baseline=float(((quantize(x)@quantize(w)-target)**2).mean()); alphas=[0.,.25,.5,.75,1.]; errors=[]; equivalence=[]
    for alpha in alphas:
        s=smooth_scales(x,w,alpha); a,b=smooth_transform(x,w,s)
        errors.append(float(((quantize(a)@quantize(b)-target)**2).mean())); equivalence.append(float((a@b-target).abs().max()))
    return {'unsmoothed_mse':baseline,'balanced_mse':errors[2],'equivalence_error':max(equivalence)}, [('INT8 reconstruction versus migration strength',alphas,[('Smoothed',errors),('Unsmoothed',[baseline]*len(alphas))])]


def run(smooth_scales, smooth_transform):
    metrics, charts = evaluate(smooth_scales, smooth_transform)
    return {'kind': 'toy_experiment', 'title': 'SmoothQuant Equivalent Activation and Weight Smoothing', 'problem': 'ai_infra/smoothquant', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2211.10438', 'anchor': 'Section 4, Equations (3) and (4); official smooth.py offline scale construction', 'adaptation': 'Use 17 samples with one activation channel amplified by 100 and correspondingly small weights; sweep alpha and compare direct and smoothed INT8 reconstruction errors.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
