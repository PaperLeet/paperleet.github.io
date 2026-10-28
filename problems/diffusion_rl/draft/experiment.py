import torch
import math


def run(draft_lv_loss):
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);noise=torch.tensor([[[-.2]],[ [.2]]],dtype=torch.float64);curve=[]
    for _ in range(55):
        generated=w.expand(1,1)
        loss=draft_lv_loss(generated,noise,.8,.6,lambda z:-w.expand_as(z),lambda z:-(z[:,0]-1).square())
        curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.08*g).detach().requires_grad_()
    metrics={'initial':curve[0],'final':curve[-1],'weight':w.item()}
    return {'kind':'toy_experiment','title':'DRaFT-LV Reward Gradients from Re-Noised Samples','problem':'diffusion_rl/draft',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2309.17400','anchor':'Section 4, DRaFT-LV and Algorithm 1 re-noising lines','adaptation':'A scalar generator and epsilon denoiser share one parameter; two fixed symmetric re-noising draws provide extra differentiable quadratic rewards over 55 updates.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
