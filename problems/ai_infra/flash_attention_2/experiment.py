import math
import torch

def evaluate(flash2_forward):
    q=torch.sin(torch.arange(52,dtype=torch.float64).reshape(13,4)*.21)*3; k=torch.cos(torch.arange(52,dtype=torch.float64).reshape(13,4)*.17)*4; v=torch.sin(torch.arange(39,dtype=torch.float64).reshape(13,3)*.3)
    sizes=[1,2,4,7,20]; errors=[]; lerrors=[]
    for size in sizes:
        e=[]; le=[]
        for causal in (False,True):
            scores=q@k.T/2
            if causal: scores=scores.masked_fill(torch.arange(13)[None,:]>torch.arange(13)[:,None],float('-inf'))
            expected=torch.softmax(scores,dim=1)@v; m=scores.max(dim=1).values; lse=m+torch.log(torch.exp(scores-m[:,None]).sum(dim=1))
            out,l=flash2_forward(q,k,v,size,3,causal); e.append(float((out-expected).abs().max())); le.append(float((l-lse).abs().max()))
        errors.append(max(e)); lerrors.append(max(le))
    return {'maximum_output_error':max(errors),'maximum_lse_error':max(lerrors)}, [('Block-size invariance',sizes,[('Output error',errors),('LSE error',lerrors)])]


def run(flash2_forward):
    metrics, charts = evaluate(flash2_forward)
    return {'kind': 'toy_experiment', 'title': 'FlashAttention-2 Delayed Normalization and Query Blocks', 'problem': 'ai_infra/flash_attention_2', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2307.08691', 'anchor': 'Section 3.1.1, Algorithm 1; Section 3.3, query-row work partitioning', 'adaptation': 'Sweep five uneven block sizes on 13 deterministic tokens, four query/key features and three value features, for causal and full attention; compare outputs and LSE with dense execution.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
