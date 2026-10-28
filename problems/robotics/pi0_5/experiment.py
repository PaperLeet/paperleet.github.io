"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='π₀.₅'
IDENTIFIER='robotics/pi0_5'
SOURCE='https://arxiv.org/abs/2504.16054'
ANCHOR='Appendix E, Figure 18: attention mask for discrete and continuous action training'
ADAPTATION='Implement Appendix E / Figure 18 training attention connectivity. Prefix, discrete FAST labels and continuous action-expert contents are supplied; omit model pretraining.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(hybrid_attention_mask):
    valid=torch.ones(1,8,dtype=torch.bool);mask=hybrid_attention_mask(valid,3,2).to(dtype=torch.float64);weights=mask/mask.sum(dim=-1,keepdim=True);values=torch.arange(8,dtype=torch.float64).reshape(1,8,1);baseline=weights@values
    leaks=[]
    for scale in [0.,1.,10.,100.,1000.]:
        changed=values+torch.tensor([0.,0.,0.,scale,scale,0.,0.,0.],dtype=values.dtype).reshape(1,8,1)
        leaks.append(((weights@changed)[:,5:]-baseline[:,5:]).abs().max().item())
    prefix=values+torch.tensor([6.,0.,0.,0.,0.,0.,0.,0.],dtype=values.dtype).reshape(1,8,1)
    return _report({'max_expert_label_leak':max(leaks),'prefix_response':((weights@prefix)[:,5:]-baseline[:,5:]).abs().mean().item(),'fast_causal_edges':int(mask[0,3,4].item())},leaks,'Expert response to FAST perturbations')
