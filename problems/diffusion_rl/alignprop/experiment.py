import torch
import math


def run(reward_rollout):
    a=torch.tensor([.4,.6,.8,.95,1.],dtype=torch.float64);s=(1-a.square()).sqrt();x=torch.zeros(8,1,dtype=torch.float64)
    w=torch.tensor(0.,dtype=torch.float64,requires_grad=True);curve=[]
    for _ in range(45):
        out=reward_rollout(x,a,s,lambda state,i: w+state*.1,1)
        loss=(out-1).square().mean();curve.append(loss.item());g,=torch.autograd.grad(loss,w);w=(w-.05*g).detach().requires_grad_()
    metrics={'initial':curve[0],'final':curve[-1],'updates':45}
    return {'kind':'toy_experiment','title':'AlignProp Truncated Reward Backpropagation','problem':'diffusion_rl/alignprop',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2310.03739','anchor':'Method: reward backpropagation; official sd_pipeline.py backprop_timestep','adaptation':'A four-step DDIM chain with a scalar epsilon bias is trained for 45 updates on a quadratic terminal target; every rollout uses the submitted truncation routine.'},
        'metrics':metrics,'charts':[{'kind':'line','title':'Measured toy result','x_label':'Step','x':list(range(len(curve))), 'series':[{'name':'Observed metric','values':curve}]}]}
