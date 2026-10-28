import torch
import math


def run(shared_noise, group_advantages, dance_loss):
    g=torch.Generator().manual_seed(42);base=torch.tensor([[-.5],[.5]],dtype=torch.float64);initial=shared_noise(base,64)[:,:,0];targets=base[:,0]+1
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[]
    for _ in range(70):
        mean=initial+w;old=mean.detach();actions=old+.6*torch.randn(old.shape,dtype=w.dtype,generator=g)
        reward=-(actions-targets[:,None]).square();rewards=torch.stack([reward,3*reward+4],-1)
        adv=group_advantages(rewards);log_ratio=(-(actions-mean).square()+(actions-old).square())/(2*.6**2)
        loss=dance_loss(log_ratio[:,:,None],adv,.2);curve.append((w.item()-1)**2)
        grad,=torch.autograd.grad(loss,w);w=(w-.03*grad).detach().requires_grad_()
    metrics={'initial_error':curve[0],'final_error':(w.item()-1)**2,'prompt_groups':2}
    return {'kind':'toy_experiment','title':'DanceGRPO Shared Initial Noise and Group Rewards','problem':'diffusion_rl/dancegrpo',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2505.07818','anchor':'Section 2.2 Eqs. (9)-(10), Initialization Noise; Section 2.3 combined rewards','adaptation':'Two prompt groups share initial states but draw independent Gaussian actions. Two differently scaled quadratic reward channels train a scalar conditional policy for 70 updates.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
