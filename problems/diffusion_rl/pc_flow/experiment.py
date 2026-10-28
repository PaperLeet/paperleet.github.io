import torch
import math


def run(pcflow_loss):
    x=torch.tensor([.5,1.,1.5,2.],dtype=torch.float64)
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[]
    for _ in range(50):
        loss=pcflow_loss(w*x,torch.zeros_like(x),-w*x,torch.zeros_like(x),.4,5)
        curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.2*g).detach().requires_grad_()
    metrics={'initial':curve[0],'final':curve[-1],'weight':w.item()}
    return {'kind':'toy_experiment','title':'PC-Flow Adjacent-Time Classifier Preference Ratios','problem':'diffusion_rl/pc_flow',
        'provenance':{'kind':'teaching_adaptation','source':'https://ojs.aaai.org/index.php/AAAI/article/view/37971','anchor':'Preference Classifier Training Objective, Eq. (10)','adaptation':'A linear classifier learns opposite next-state changes for winner and loser transitions over 50 actual updates.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
