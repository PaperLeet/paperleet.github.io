import torch

def evaluate(gptq_quantize):
    x=torch.tensor([[1.,-.8],[0.,.6]],dtype=torch.float64); w=torch.tensor([[.49,.7],[-.49,-.7]],dtype=torch.float64)
    # X^T X has inverse [[25/9,20/9],[20/9,25/9]], factored as U^T U.
    u=torch.tensor([[5/3,4/3],[0.,1.]],dtype=torch.float64); steps=[.25,.4,.6,.8,1.]; corrected=[]; naive=[]
    for step in steps:
        q=gptq_quantize(w,u,torch.full((2,),step,dtype=torch.float64),-7,7)
        rtn=torch.round(w/step).clamp(-7,7)*step
        corrected.append(float(((x@(q-w).T)**2).mean())); naive.append(float(((x@(rtn-w).T)**2).mean()))
    return {'compensated_mse':corrected[-1],'rtn_mse':naive[-1]}, [('Correlated calibration reconstruction',steps,[('GPTQ',corrected),('Round to nearest',naive)])]


def run(gptq_quantize):
    metrics, charts = evaluate(gptq_quantize)
    return {'kind': 'toy_experiment', 'title': 'GPTQ Cholesky Error-Compensated Quantization', 'problem': 'ai_infra/gptq', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2210.17323', 'anchor': 'Section 4, Algorithm 1; official gptq.py fasterquant inner column loop', 'adaptation': 'Use two correlated calibration features and two weight rows, sweep fixed quantization steps, and compare output-reconstruction MSE against independent round-to-nearest.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
