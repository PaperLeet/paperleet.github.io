"""Deterministic categorical policy learning on fixed toy feedback; no LLM generation."""
import torch

def run(impl, steps=60):
    dtype=torch.float64
    a=torch.tensor([[-1.,-1.],[-1.,-1.],[-1.,-1.],[-1.,-1.]],dtype=dtype)
    b=torch.tensor([[-2.,-2.],[-1.2,-1.2],[-1.4,-1.4],[-1.9,-1.9]],dtype=dtype)
    delta=impl.hint_delta(a,b,torch.ones(4,2,dtype=torch.bool))
    rewards=impl.proposer_rewards(delta,torch.tensor([200,200,200,200]),torch.full((4,),.25,dtype=dtype),torch.tensor([True,True,True,False]))
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
