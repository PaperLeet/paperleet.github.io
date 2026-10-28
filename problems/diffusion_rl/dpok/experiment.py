import torch
import math


def run(gaussian_kl, dpok_surrogate):
    g=torch.Generator().manual_seed(42);w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[];beta=.4
    for _ in range(70):
        curve.append((w.item()-2/(2+beta))**2)
        mean=w.expand(256,1,1);actions=mean.detach()+torch.randn((256,1,1),dtype=w.dtype,generator=g)
        reward=-(actions[:,0,0]-1).square();reward=reward-reward.mean()
        logp=-.5*(actions[:,:,0]-mean[:,:,0]).square()
        kl=gaussian_kl(mean,torch.zeros_like(mean),w.new_tensor([1.]))
        loss=dpok_surrogate(logp,reward,kl,1.,beta);grad,=torch.autograd.grad(loss,w);w=(w-.035*grad).detach().requires_grad_()
    metrics={'initial_error':curve[0],'final_error':(w.item()-2/(2+beta))**2,'mean':w.item()}
    return {'kind':'toy_experiment','title':'DPOK Gaussian KL Bound and Policy Gradient','problem':'diffusion_rl/dpok',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2305.16381','anchor':'Section 4.1 Lemma 4.2 and Eqs. (7)-(9), efficient gradient approximation','adaptation':'A Gaussian scalar policy with variance 1 learns a quadratic target under beta=0.4 KL to a zero-mean reference; 70 updates are compared against the analytic regularized optimum 2/(2+beta).'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
