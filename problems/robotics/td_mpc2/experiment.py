"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='TD-MPC2'
IDENTIFIER='robotics/td_mpc2'
SOURCE='https://arxiv.org/abs/2310.16828'
ANCHOR='SimNorm and official _plan elite weighted mean/std update'
ADAPTATION='Isolate SimNorm and MPPI weighted moments for already-selected elites. Candidate trajectories and values are supplied; omit learned dynamics/Q networks and real MPC.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(simnorm,elite_moments):
    mean=torch.zeros(2,1,dtype=torch.float64);std=torch.ones_like(mean);grid=torch.tensor([[a,b] for a in [-2.,-1.,0.,1.,2.] for b in [-2.,-1.,0.,1.,2.]],dtype=mean.dtype).transpose(0,1).unsqueeze(-1);target=mean.new_tensor([[.7],[-.4]]);curve=[]
    for _ in range(12):
        actions=mean.unsqueeze(1)+std.unsqueeze(1)*grid;values=-(actions-target.unsqueeze(1)).square().sum(dim=0).sum(dim=-1)
        order=sorted(range(25),key=lambda i:float(values[i]),reverse=True)[:8];elite=actions[:,order];score=values[order];mean,std=elite_moments(elite,score,5.,.03,1.);curve.append((mean-target).square().mean().item())
    latent=simnorm(torch.tensor([[1.,2.,3.,4.,5.,6.]],dtype=mean.dtype),3).reshape(1,2,3)
    return _report({'initial_error':.5*(.7**2+.4**2),'final_error':curve[-1],'simplex_error':(latent.sum(-1)-1).abs().max().item()},curve,'Elite action-sequence refinement')
