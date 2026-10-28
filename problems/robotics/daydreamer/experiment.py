"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='DayDreamer'
IDENTIFIER='robotics/daydreamer'
SOURCE='https://arxiv.org/abs/2206.14176'
ANCHOR='Appendix D: DreamerV2 world model, KL balancing 0.8'
ADAPTATION='Isolate the DreamerV2-style KL balancing used by DayDreamer, with default alpha=0.8. Compute KL before free-nats clipping; omit the RSSM and online robot training.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(balanced_kl):
    q=torch.tensor([[[.7,.2,.1],[.1,.3,.6]]],dtype=torch.float64).log();prior=torch.zeros_like(q,requires_grad=True);curve=[]
    for i in range(81):
        loss=balanced_kl(q,prior,.8);curve.append(loss.item())
        if i<80:g,=torch.autograd.grad(loss,prior);prior=(prior-g).detach().requires_grad_()
    return _report({'initial_kl':curve[0],'final_kl':curve[-1],'probability_error':(torch.softmax(prior,-1)-q.exp()).abs().max().item()},curve,'Prior fitting under balanced KL')
