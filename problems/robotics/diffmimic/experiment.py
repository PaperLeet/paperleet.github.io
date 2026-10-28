"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='DiffMimic'
IDENTIFIER='robotics/diffmimic'
SOURCE='https://arxiv.org/abs/2304.03274'
ANCHOR='Equations 3.1, 3.3, 3.5 and Algorithm 1: PD torque, state matching and demonstration replay'
ADAPTATION='Implement error-triggered demonstration replay, PD torque and temporal state matching. The experiment uses a scalar differentiable integrator, not a Brax humanoid reproduction.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(replay_state,pd_torque,state_matching_loss):
    initial=torch.tensor([[-1.],[.4],[1.2]],dtype=torch.float64);reference=torch.zeros_like(initial);gain=torch.tensor(1.,dtype=initial.dtype,requires_grad=True);curve=[];replays=0
    for iteration in range(61):
        q=initial;v=torch.zeros_like(q);states=[]
        for t in range(12):
            states.append(q);state=torch.cat((q,v),dim=-1);ref=torch.zeros_like(state);selected=replay_state(state,ref,2.);replays+=int((selected-state).abs().sum().item()>0)
            q=selected[:,:1];v=selected[:,1:];target=gain*q;torque=pd_torque(q,v,target,2.,.4);v=v+.15*torque;q=q+.15*v
        trajectory=torch.stack(states,dim=1);loss=state_matching_loss(trajectory,torch.zeros_like(trajectory));curve.append(loss.item())
        if iteration<60:g,=torch.autograd.grad(loss,gain);gain=(gain-.04*g).detach().requires_grad_()
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'gain':gain.item(),'replay_probe':replay_state(initial,reference,.1).abs().sum().item()},curve,'Differentiable PD trajectory fitting')
