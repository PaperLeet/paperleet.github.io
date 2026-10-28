import torch

def evaluate(group_quantize,awq_search):
    x=torch.sin(torch.arange(64,dtype=torch.float64).reshape(16,4)*.43)*torch.tensor([8.,1.,1.,1.],dtype=torch.float64)
    w=torch.cos(torch.arange(12,dtype=torch.float64).reshape(3,4)*.71)*.8; alphas=[i/10 for i in range(11)]; scores=[]
    for a in alphas:
        _,q,_,_=awq_search(x,w,3,4,[a]); scores.append(float(((x@q.T-x@w.T)**2).mean()))
    scales,chosen,a,reported=awq_search(x,w,3,4,alphas); actual=float(((x@chosen.T-x@w.T)**2).mean())
    baseline=float(((x@group_quantize(w,3,4).T-x@w.T)**2).mean())
    return {'best_mse':actual,'baseline_mse':baseline,'chosen_alpha':a,'reported_error_difference':abs(reported-actual)}, [('AWQ exponent calibration',alphas,[('Output reconstruction MSE',scores),('No scaling',[baseline]*len(alphas))])]


def run(group_quantize, awq_search):
    metrics, charts = evaluate(group_quantize, awq_search)
    return {'kind': 'toy_experiment', 'title': 'AWQ Activation-Aware Scaling Calibration', 'problem': 'ai_infra/awq', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2306.00978', 'anchor': 'Section 3.2: activation-aware scaling search; official auto_scale.py _search_module_scale', 'adaptation': 'Use 16 deterministic calibration samples with one channel amplified eightfold and 3×4 weights under three-bit group quantization. Sweep eleven alpha values from zero to one and compare the selected candidate with alpha zero.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
