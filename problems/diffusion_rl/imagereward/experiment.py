import torch
import math


def run(ranking_loss, refl_clean):
    x=torch.linspace(.2,1.6,8,dtype=torch.float64)
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    curve=[]
    for _ in range(40):
        loss=ranking_loss(w*x,-w*x);curve.append(loss.item())
        g,=torch.autograd.grad(loss,w);w=(w-.3*g).detach().requires_grad_()
    initial=curve[0]; ranking_final=curve[-1]
    e=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    frozen_w=w.detach();latent=torch.zeros(8,1,dtype=torch.float64)
    rewards=[]
    for _ in range(30):
        clean=refl_clean(latent,e.expand(8,1),.8,.6)
        reward=(frozen_w*clean).mean();rewards.append(reward.item())
        g,=torch.autograd.grad(-reward,e);e=(e-.03*g).detach().requires_grad_()
    metrics={'initial':initial,'final':ranking_final,'reward_gain':rewards[-1]-rewards[0]}
    return {'kind':'toy_experiment','title':'ImageReward Ranking and ReFL One-Step Feedback','problem':'diffusion_rl/imagereward',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2304.05977','anchor':'Section 2.2 Eq. (1); Section 3 Algorithm 1','adaptation':'Eight scalar preference pairs train a linear reward scorer; 30 subsequent reward-gradient steps update a one-step epsilon model. This tests feedback mechanics, not image quality.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
