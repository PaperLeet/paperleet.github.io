import torch
import math


def run(window_start, mixed_transition, window_loss):
    g=torch.Generator().manual_seed(42);w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);times=[.8,.6,.4,.2,0.];curve=[]
    for update in range(60):
        start=window_start(update,4,2,10,1);state=torch.zeros(128,1,dtype=w.dtype);ratios=[]
        for i in range(4):
            mean,std=mixed_transition(state,w.expand_as(state),times[i],times[i+1],.5,start<=i<start+2)
            old=mean.detach();action=old+std*torch.randn(old.shape,dtype=w.dtype,generator=g)
            ratios.append(((-(action-mean).square()+(action-old).square())/(2*std.square())).sum(-1) if std.item()>0 else w.expand(128)*0)
            state=action
        reward=-(state[:,0]-1).square();adv=reward-reward.mean()
        loss=window_loss(torch.stack(ratios,1),adv,start,2,.2);grad,=torch.autograd.grad(loss,w);w=(w-.15*grad).detach().requires_grad_()
        curve.append((state.mean().item()-1)**2)
    metrics={'early_error':sum(curve[:5])/5,'late_error':sum(curve[-5:])/5,'last_start':start}
    return {'kind':'toy_experiment','title':'MixGRPO Sliding Stochastic Windows and Local Policy Objectives','problem':'diffusion_rl/mixgrpo',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2507.21802','anchor':'Section 3.1 mixed ODE-SDE; Algorithm 1; Section 3.2 Eq. (11)','adaptation':'A four-transition scalar flow uses a moving two-step stochastic window. Fresh Gaussian paths and terminal quadratic rewards update one velocity parameter for 60 iterations.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
