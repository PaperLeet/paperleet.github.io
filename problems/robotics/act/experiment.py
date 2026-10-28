"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='ACT'
IDENTIFIER='robotics/act'
SOURCE='https://arxiv.org/abs/2304.13705'
ANCHOR='Temporal ensembling at inference; official imitate_episodes.py exp_weights and CVAE policy KL'
ADAPTATION='Implement two ACT tensor cores with predictions already aligned to one execution time; omit the visual Transformer, action-chunk network and teleoperation system.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(temporal_ensemble,style_kl):
    target=torch.sin(torch.arange(64,dtype=torch.float64)*.15);offset=torch.tensor([1.,-1.,1.,-1.,1.],dtype=target.dtype)
    raw=[];smooth=[]
    for value in target:
        candidates=(value+.3*offset).reshape(5,1)
        raw.append(float((candidates[-1,0]-value).square()))
        smooth.append(float((temporal_ensemble(candidates,torch.ones(5,dtype=torch.bool),.01)[0]-value).square()))
    mu=torch.ones(4,3,dtype=target.dtype,requires_grad=True);logs=torch.zeros_like(mu);kl=[]
    for step in range(41):
        loss=style_kl(mu,logs);kl.append(loss.item())
        if step<40: g,=torch.autograd.grad(loss,mu);mu=(mu-.5*g).detach().requires_grad_()
    return _report({'latest_mse':sum(raw)/64,'ensemble_mse':sum(smooth)/64,'initial_kl':kl[0],'final_kl':kl[-1]},kl,'Style KL')
