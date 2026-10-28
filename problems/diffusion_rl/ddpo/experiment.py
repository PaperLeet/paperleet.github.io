import torch
import math


def run(gaussian_log_prob, ddpo_loss):
    g=torch.Generator().manual_seed(42);w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);s=w.new_tensor([.6]);curve=[]
    for _ in range(60):
        curve.append((w.item()-1)**2)
        mean=w.expand(128,1,1);a=mean.detach()+.6*torch.randn((128,1,1),dtype=w.dtype,generator=g)
        logp=gaussian_log_prob(mean,a,s);reward=-(a[:,0,0]-1).square();adv=reward-reward.mean()
        loss=ddpo_loss(logp,logp.detach(),adv,.2);grad,=torch.autograd.grad(loss,w);w=(w-.08*grad).detach().requires_grad_()
    metrics={'initial_error':curve[0],'final_error':(w.item()-1)**2,'updates':60}
    return {'kind':'toy_experiment','title':'DDPO Gaussian Transition Densities and Stepwise PPO','problem':'diffusion_rl/ddpo',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2305.13301','anchor':'Section 4.3 Eq. (6), importance-sampled DDPO; Appendix D.1','adaptation':'A scalar Gaussian policy with standard deviation 0.6 draws 128 fresh actions per update; 60 PPO gradient updates optimize a quadratic terminal reward toward target 1.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
