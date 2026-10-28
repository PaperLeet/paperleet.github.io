import torch
import math


def run(d3po_loss):
    a=torch.tensor([[1.,-.5,2.],[.5,-2.,1.]],dtype=torch.float64)
    w=torch.zeros(3,dtype=torch.float64,requires_grad=True);curve=[]
    for _ in range(60):
        p=w[None,:]*a.abs();loss=d3po_loss(p,-p,torch.zeros_like(p),torch.zeros_like(p),1.)
        curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.3*g).detach().requires_grad_()
    metrics={'initial':curve[0],'final':curve[-1],'minimum_margin':w.min().item()}
    return {'kind':'toy_experiment','title':'D3PO Per-Transition Preference Optimization','problem':'diffusion_rl/d3po',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2311.13231','anchor':'Section 4.3 Eq. (14): a preference objective for every state-action pair','adaptation':'A three-step preference model learns independently signed transition margins through 60 gradient updates.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
