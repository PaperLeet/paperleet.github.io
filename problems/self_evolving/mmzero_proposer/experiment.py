"""Deterministic categorical policy learning on fixed toy feedback; no LLM generation."""
import torch

def run(impl, steps=60):
    dtype=torch.float64
    ex=torch.tensor([[True,True],[True,False],[False,False],[True,True]])
    easy=torch.ones(4,2,dtype=dtype)
    c=torch.tensor([[.5,.5],[.5,.5],[.5,.5],[1.,1.]],dtype=dtype)
    share=torch.tensor([[.25,.25,.25],[.25,.25,.25],[.25,.25,.25],[1.,1.,1.]],dtype=dtype)
    div=impl.diversity_adjustment(share,4)
    adjustments=torch.stack([torch.zeros_like(div),torch.zeros_like(div),div],dim=1)
    rewards=impl.proposal_rewards(ex,easy,c,adjustments,torch.ones(4,dtype=torch.bool))
    target=0
    logits=torch.zeros(rewards.shape[0],dtype=dtype,requires_grad=True)
    curve=[]
    for step in range(steps+1):
        probabilities=torch.softmax(logits,dim=0)
        curve.append(float(probabilities[target].detach()))
        if step==steps:break
        loss=-(probabilities*rewards.detach()).sum()
        loss.backward()
        with torch.no_grad():
            logits-=.5*logits.grad
        logits.grad=None
    return {'metrics':{'initial_probability':curve[0],'final_probability':curve[-1]},
            'charts':[{'kind':'line','title':'Toy policy: probability of target action',
            'x':list(range(steps+1)),'series':[{'name':'Target action probability','values':curve}]}]}
