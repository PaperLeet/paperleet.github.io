"""Deterministic categorical policy learning on fixed toy feedback; no LLM generation."""
import torch

def run(impl, steps=60):
    dtype=torch.float64
    records=[dict(question='q'+str(i),hint='secret',unassisted='1',assisted='0') for i in range(4)]
    pairs=impl.build_preferences(records,torch.tensor([-.2,.1,.7,2.],dtype=dtype),torch.tensor([True,True,True,True]))
    rewards=torch.zeros(2,dtype=dtype)
    for pair in pairs:
        rewards[int(pair['chosen'])]+=1./2
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
