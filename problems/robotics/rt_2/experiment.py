"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='RT-2'
IDENTIFIER='robotics/rt_2'
SOURCE='https://arxiv.org/abs/2307.15818'
ANCHOR='Section 3.2 Output Constraint'
ADAPTATION='Implement output constraints for robot and vision-language rows sharing logits; callers supply action-token identities. Omit PaLI-X, PaLM-E and semantic reasoning.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(action_probabilities):
    z=torch.zeros(2,4,dtype=torch.float64,requires_grad=True);allowed=torch.tensor([False,True,False,True]);robot=torch.tensor([True,False]);curve=[]
    for i in range(81):
        p=action_probabilities(z,allowed,robot);loss=-torch.log(p[:,3]).mean();curve.append(loss.item())
        if i<80:g,=torch.autograd.grad(loss,z);z=(z-g).detach().requires_grad_()
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'forbidden_robot_mass':p[0,~allowed].sum().item(),'vlm_mass':p[1,~allowed].sum().item()},curve,'Constrained target negative log probability')
