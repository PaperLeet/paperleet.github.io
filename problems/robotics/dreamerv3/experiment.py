"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='DreamerV3'
IDENTIFIER='robotics/dreamerv3'
SOURCE='https://arxiv.org/abs/2301.04104'
ANCHOR='Symlog predictions and two-hot distribution; Appendix Distributions'
ADAPTATION='Implement scale-robust scalar transforms and discrete-support encoding; omit the RSSM, imagined policy and Minecraft. two_hot receives values already in symlog space and caller-supplied centers.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(symlog,symexp,two_hot):
    raw=torch.tensor([-100.,-9.,-.5,.5,9.,100.],dtype=torch.float64);bins=torch.linspace(-6,6,25,dtype=raw.dtype);labels=two_hot(symlog(raw),bins)
    reconstructed=symexp((labels*bins).sum(-1));logits=torch.zeros_like(labels,requires_grad=True);curve=[]
    for i in range(151):
        loss=-(labels*torch.log_softmax(logits,dim=-1)).sum(-1).mean();curve.append(loss.item())
        if i<150:g,=torch.autograd.grad(loss,logits);logits=(logits-3*g).detach().requires_grad_()
    kl=(labels*(torch.log(labels.clamp_min(1e-30))-torch.log_softmax(logits,dim=-1))).sum(-1).mean()
    return _report({'initial_cross_entropy':curve[0],'final_cross_entropy':curve[-1],'final_kl':kl.item(),'roundtrip_error':(raw-reconstructed).abs().max().item()},curve,'Transformed categorical value fitting')
