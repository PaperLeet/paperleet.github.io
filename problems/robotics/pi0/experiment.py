"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='π₀'
IDENTIFIER='robotics/pi0'
SOURCE='https://arxiv.org/abs/2410.24164'
ANCHOR='Official openpi Pi0.compute_loss and sample_actions time orientation'
ADAPTATION='Isolate the official openpi linear flow path and Euler sampling direction; explicit noise/time and a linear velocity regressor replace the VLA action expert.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(flow_interpolate,euler_step):
    a=torch.linspace(-1,1,12,dtype=torch.float64).reshape(6,2,1);noise=-a;time=torch.linspace(0,1,6,dtype=a.dtype);_,target=flow_interpolate(a,noise,time)
    gain=torch.tensor(0.,dtype=a.dtype,requires_grad=True);curve=[]
    for i in range(61):
        loss=(gain*a-target).square().mean();curve.append(loss.item())
        if i<60:g,=torch.autograd.grad(loss,gain);gain=(gain-.5*g).detach().requires_grad_()
    s=noise
    for _ in range(10):s=euler_step(s,gain.detach()*a,-.1)
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'sample_mse':(s-a).square().mean().item(),'gain':gain.item()},curve,'Velocity regression')
