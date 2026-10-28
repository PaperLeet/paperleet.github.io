"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='DeepMimic'
IDENTIFIER='robotics/deepmimic'
SOURCE='https://arxiv.org/abs/1804.02717'
ANCHOR='Section 5.3 Reward Function: weighted pose, velocity, end-effector and center-of-mass imitation terms'
ADAPTATION='Implement pose, angular-velocity, end-effector and center-of-mass imitation rewards. Pose inputs are precomputed quaternion geodesic angle errors; omit PPO, pose parsing and physical simulation.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(imitation_reward):
    scales=torch.linspace(1,0,31,dtype=torch.float64);pose=scales[:,None]*torch.tensor([[.3,.5]],dtype=scales.dtype);velocity=scales[:,None,None]*torch.ones(1,2,3,dtype=scales.dtype)*.4;ends=scales[:,None,None]*torch.ones(1,2,3,dtype=scales.dtype)*.08;com=scales[:,None]*torch.ones(1,3,dtype=scales.dtype)*.1
    rewards=imitation_reward(pose,velocity,ends,com);curve=rewards.tolist()
    return _report({'initial_reward':curve[0],'perfect_reward':curve[-1],'monotone_steps':int((rewards[1:]>rewards[:-1]).sum().item()),'bounded':bool(((rewards>0)&(rewards<=1)).all())},curve,'Reward along progressively closer motions')
