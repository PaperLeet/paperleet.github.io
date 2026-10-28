"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='RDT-1B'
IDENTIFIER='robotics/rdt_1b'
SOURCE='https://arxiv.org/abs/2410.07864'
ANCHOR='Physically interpretable unified action space; official RDTRunner.conditional_sample final action mask'
ADAPTATION='Isolate unified action placement and final sampling masks using a configurable small slot map, not the complete 128D schema or 1.2B Transformer.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(pack_actions,mask_actions):
    target=torch.tensor([[[.2,-.4,.8]]],dtype=torch.float64);unified,mask=pack_actions(target,[4,0,2],6)
    p=torch.full((1,1,6),9.,dtype=target.dtype,requires_grad=True);curve=[]
    for step in range(101):
        executed=mask_actions(p,mask);loss=(executed-unified).square().mean();curve.append(loss.item())
        if step<100:g,=torch.autograd.grad(loss,p);p=(p-.5*g).detach().requires_grad_()
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'invalid_max':executed[:,:,~mask].abs().max().item()},curve,'Valid action fit')
