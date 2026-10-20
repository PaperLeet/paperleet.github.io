"""Deterministic categorical policy learning on fixed toy feedback; no LLM generation."""
import torch

def run(impl, steps=60):
    dtype=torch.float64
    answers=torch.tensor([[0,0,0,1],[1,1,1,1]])
    labels,shares=impl.consensus(answers,2)
    keep=impl.curriculum_mask(shares)
    candidates=torch.tensor([[0,1],[0,1]])
    rewards=(impl.pseudo_rewards(candidates,labels)*keep[:,None]).sum(dim=0)
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
