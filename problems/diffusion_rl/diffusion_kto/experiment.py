import torch
import math


def run(kto_loss):
    y=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[]
    for _ in range(60):
        loss=kto_loss(w*y,torch.zeros_like(y),y,y.new_tensor([-.3,.1]),2.)
        curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.2*g).detach().requires_grad_()
    metrics={'initial':curve[0],'final':curve[-1],'margin':w.item()}
    return {'kind':'toy_experiment','title':'Diffusion-KTO Binary Feedback and Utility Baseline','problem':'diffusion_rl/diffusion_kto',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2404.04465','anchor':'Section 4.1 Eqs. (7)-(8), empirical nonnegative unrelated-pair baseline','adaptation':'A scalar policy preference offset is optimized on balanced desirable/undesirable labels for 60 steps with a separate fixed baseline batch.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
