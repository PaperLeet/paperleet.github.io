import math
import torch

def evaluate(parallel_mlp):
    x=torch.sin(torch.arange(27,dtype=torch.float64).reshape(9,3)*.3); a=torch.cos(torch.arange(24,dtype=torch.float64).reshape(3,8)*.21)*.4
    c=torch.linspace(-.2,.2,8,dtype=torch.float64); d=torch.tensor([.1,-.1],dtype=torch.float64)
    z=x@a+c; hidden=.5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3)))
    target=hidden@torch.sin(torch.arange(16,dtype=torch.float64).reshape(8,2)) + d
    w=torch.zeros(8,2,dtype=torch.float64); dense=w.clone(); losses=[]; errors=[]
    for _ in range(30):
        w=w.detach().requires_grad_(); pred=parallel_mlp(x,a,c,w,d,2); loss=((pred-target)**2).mean(); grad,=torch.autograd.grad(loss,(w,))
        w=w-1.0*grad; dense=dense-1.0*(2*hidden.T@(hidden@dense+d-target)/target.numel())
        losses.append(float(loss.detach())); errors.append(float((w-dense).abs().max()))
    return {'initial_loss':losses[0],'final_loss':losses[-1],'trajectory_error':max(errors)}, [('Tensor-parallel SGD',list(range(30)),[('MSE',losses),('Dense weight difference',errors)])]


def run(parallel_mlp):
    metrics, charts = evaluate(parallel_mlp)
    return {'kind': 'toy_experiment', 'title': 'Megatron Column- and Row-Parallel MLP', 'problem': 'ai_infra/megatron_lm', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1909.08053', 'anchor': 'Section 3, Figure 3: column-partitioned first GEMM and row-partitioned second GEMM', 'adaptation': 'Use fixed 9×3 inputs and eight hidden channels; train output weights with two shards for 30 steps and compare loss, weights and gradients against dense execution.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
