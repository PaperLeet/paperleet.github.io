import torch
import math


def run(policy_kl, flow_dppo_loss):
    g=torch.Generator().manual_seed(42);w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[];std=w.new_tensor([.7])
    for _ in range(65):
        mean=w.expand(128,1,1);old=mean.detach();actions=old+.7*torch.randn(old.shape,dtype=w.dtype,generator=g)
        reward=-(actions[:,0,0]-1).square();adv=reward-reward.mean()
        lr=((actions-old).square()-(actions-mean).square()).sum(-1)/(2*.7**2)
        oldkl=policy_kl(mean,old,std);refkl=policy_kl(mean,torch.zeros_like(mean),std)
        loss=flow_dppo_loss(lr,adv,oldkl,refkl,.02,.1);curve.append((w.item()-1/(1+.1/(2*.7**2)))**2)
        grad,=torch.autograd.grad(loss,w);w=(w-.06*grad).detach().requires_grad_()
    metrics={'initial_error':curve[0],'final_error':curve[-1],'updates':65}
    return {'kind':'toy_experiment','title':'Flow-DPPO Exact Divergence and Directional Masks','problem':'diffusion_rl/flow_dppo',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2606.11025','anchor':'Section 3.3 Eqs. (14), (17)-(18): Gaussian divergence and directional mask','adaptation':'A scalar Gaussian policy draws 128 fresh actions per update and optimizes a quadratic reward with beta=0.1 reference KL for 65 iterations. Independent tests exercise nonzero old-policy divergence masks.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
