"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='Octo'
IDENTIFIER='robotics/octo'
SOURCE='https://arxiv.org/abs/2405.12213'
ANCHOR='Official action_heads.py masked_mean and timestep/action pad-mask intersection'
ADAPTATION='Isolate the action head masked_mean and intersection of timestep/action padding. Compute coordinatewise squared error before any head-specific action_dim scaling; omit the diffusion network.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(masked_action_loss):
    p=torch.zeros(1,2,2,2,dtype=torch.float64,requires_grad=True);target=torch.tensor([[[[1.,100.],[2.,100.]],[[100.,100.],[100.,100.]]]],dtype=p.dtype)
    t=torch.tensor([[True,False]]);a=torch.tensor([[[[True,False],[True,False]],[[True,True],[True,True]]]]);curve=[]
    for i in range(41):
        loss=masked_action_loss(p,target,t,a);curve.append(loss.item())
        if i<40:g,=torch.autograd.grad(loss,p);p=(p-.2*g).detach().requires_grad_()
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'padded_motion':p[0,1].abs().sum().item()+p[0,0,:,1].abs().sum().item()},curve,'Masked action fitting')
